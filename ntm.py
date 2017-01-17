# the core part of our neural turing machine
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tesorflow as tf

#from util import
from ops import linear, Linear, softmax

class NTMTracker(object):
    def __init__(self, num_features, feature_depth, max_sequence_length=300, mem_size=128, mem_dim=20,
            controller_dim=100, controller_layer_size=1,
            write_head_size=1, read_head_size=1):
        """
        The ntm tracker core.

        Args:
            num_features: number of deep feature vectors NTM needs to select
            from. This parameter determines the dimension of final output of
            controller, and the additional input dimension expected

            feature_depth: number of channels of the features. Currently only
            look at fatures of the same depth.

            max_sequence_length: the maximum value of sequence length. The graph
            will have the same length.
        """
        self.num_fatures = num_features
        self.feature_depth = feature_depth
        self.max_sequence_length = max_sequence_length
        self.mem_size = mem_size
        self.mem_dim = mem_dim
        self.controller_dim = controller_dim
        self.controller_layer_size = controller_layer_size
        self.write_head_size = write_head_size
        self.read_head_size = read_head_size

    def __call__(self, inputs, target, state=None, scope=None):
        """
        Args:
            input: Input should be extracted features of VGG network.
            For current stage, we only look at invariable scale objects in video
            sequence. That means we only need one layer feature map as input.
            The input will be serialized into a 2D-vector, [num_features x channels]
            The inputs is the stack of the 2D vector, [steps x num_features x channels]

            target: A target indicator for the first frame is also given.
            It will be a 1D vector of num_features size, 1 for positive features,
            -1 for negative features. On subsequent frames, 0 vector will be
            given.

            state: state dictionary which contains M, read_w, write_w, read,
            output, hidden.

            scope: VariableScope for the created subgraph; defaults to class
            name.

        1. build the controller chain
        2. build the memory module
        """
        for idx in xrange(self.max_sequence_length):
            if idx == 0:
                # initial state for the first frame
                _, state = self.initial_state()
                indicator = tf.reshape(target, [-1]) # [num_features]
            else:
                tf.get_variable_scope().reuse_variables()
                indicator = tf.constant(0, shape=[self.num_features])

            M_prev = state['M'] #memory value of previous timestep
            read_w_list_prev = state['read_w'] #read weight history
            write_w_list_prev = state['write_w'] #write weight history
            read_list_prev = state['read'] #read value history
            #last output. It's a list because controller is an LSTM with multiple cells
            output_list_prev = state['output']
            hidden_list_prev = state['hidden'] #hidden state history

            output_list, hidden_list = self.build_controller(inputs[idx],
                    indicator, read_list_prev, output_list_prev,
                    hidden_list_prev)

            # last output layer from LSTM controller
            last_output = output_list[-1]

            # build a memory
            M, read_w_list, write_w_list, read_list = self.build_memory(M_prev,
                    read_w_list_prev, write_w_list_prev, last_output)

    def build_memory(self, M_prev, read_w_list_prev, write_w_list_prev, last_output):
        """Build a memory to read & write."""

        with tf.variable_scope("memory"):
            # 3.1 Reading
            read_w_list = []
            read_list = []
            for idx in xrange(self.read_head_size):
                read_w_prev = read_w_list_prev[idx]
                read_w, read = self.build_read_head(M_prev,
                        tf.squeeze(read_w_prev), last_output, idx)

                read_w_list.append(read_w)
                read_list.append(read)

            # 3.2 Writing
            if self.write_head_size == 1:
                write_w_prev = write_w_list_prev[0]

                write_w, write, erase = self.build_write_head(M_prev,
                                                              tf.squeeze(write_w_prev),
                                                              last_output, 0)

                M_erase = tf.ones([self.mem_size, self.mem_dim]) \
                                  - outer_product(write_w, erase)
                M_write = outer_product(write_w, write)

                write_w_list = [write_w]
            else:
                write_w_list = []
                write_list = []
                erase_list = []

                M_erases = []
                M_writes = []

                for idx in xrange(self.write_head_size):
                    write_w_prev_idx = write_w_list_prev[idx]

                    write_w_idx, write_idx, erase_idx = \
                        self.build_write_head(M_prev, write_w_prev_idx,
                                              last_output, idx)

                    write_w_list.append(tf.transpose(write_w_idx))
                    write_list.append(write_idx)
                    erase_list.append(erase_idx)

                    M_erases.append(tf.ones([self.mem_size, self.mem_dim]) \
                                    - outer_product(write_w_idx, erase_idx))
                    M_writes.append(outer_product(write_w_idx, write_idx))

                M_erase = reduce(lambda x, y: x*y, M_erases)
                M_write = tf.add_n(M_writes)

            M = M_prev * M_erase + M_write

            return M, read_w_list, write_w_list, read_list

    def build_controller(self, input_, target, read_list_prev, output_list_prev,
            hidden_list_prev):
        #build the controller
        output_list = []
        hidden_list = []
        with tf.variable_scope("controller"):
            # for every layer of the lstm
            for layer_idx in xrange(self.controller_laye_size):
                o_prev = output_list_prev[layer_idx] # [num_features, 1]
                h_prev = hidden_list_prev[layer_idx]

                if layer_idx == 0: #the first layer
                    # gate input is input, previous output, value read from mem
                    def new_gate(gate_name):
                        #             [#f*#c], [#f],   [#f],   [#f, 1]xn
                        return linear([input_, target, o_prev] + read_list_prev,
                                      output_size = self.controller_dim,
                                      bias = True,
                                      scope = "%s_gate_%s" % (gate_name, layer_idx))
                        # output will be [num_features, controller_dim]
                else:
                    # gate input is output from lower layer and previous output
                    def new_gate(gate_name):
                        return linear([output_list[-1], o_prev],
                                      output_size = self.controller_dim,
                                      bias = True,
                                      scope="%s_gate_%s" % (gate_name, layer_idx))

                # input, forget, and output gates for LSTM
                i = tf.sigmoid(new_gate('input'))
                f = tf.sigmoid(new_gate('forget'))
                o = tf.sigmoid(new_gate('output'))
                update = tf.tanh(new_gate('update'))

                # update the sate of the LSTM cell
                hid = tf.add_n([f * h_prev, i * update])
                out = o * tf.tanh(hid)

                hidden_list.append(hid)
                output_list.append(out)

        return output_list, hidden_list

    def build_read_head(self, M_prev, read_w_prev, last_output, idx):
        return self.build_head(M_prev, read_w_prev, last_output, True, idx)

    def build_write_head(self, M_prev, write_w_prev, last_output, idx):
        return self.build_head(M_prev, write_w_prev, last_output, False, idx)

    def build_head(self, M_prev, w_prev, last_output, is_read, idx):
        scope = "read" if is_read else "write"

        with tf.variable_scope(scope):
            # Figure 2.
            # Amplify or attenuate the precision
            with tf.variable_scope("k"):
                k = tf.tanh(Linear(last_output, self.mem_dim, name='k_%s' % idx))
            # Interpolation gate
            with tf.variable_scope("g"):
                g = tf.sigmoid(Linear(last_output, 1, name='g_%s' % idx))
            # shift weighting
            with tf.variable_scope("s_w"):
                w = Linear(last_output, 2 * self.shift_range + 1, name='s_w_%s' % idx)
                s_w = softmax(w)
            with tf.variable_scope("beta"):
                beta  = tf.nn.softplus(Linear(last_output, 1, name='beta_%s' % idx))
            with tf.variable_scope("gamma"):
                gamma = tf.add(tf.nn.softplus(Linear(last_output, 1, name='gamma_%s' % idx)),
                               tf.constant(1.0))

            # 3.3.1
            # Cosine similarity
            similarity = smooth_cosine_similarity(M_prev, k) # [mem_size x 1]
            # Focusing by content
            content_focused_w = softmax(scalar_mul(similarity, beta))

            # 3.3.2
            # Focusing by location
            gated_w = tf.add_n([
                scalar_mul(content_focused_w, g),
                scalar_mul(w_prev, (tf.constant(1.0) - g))
            ])

            # Convolutional shifts
            conv_w = circular_convolution(gated_w, s_w)

            # Sharpening
            powed_conv_w = tf.pow(conv_w, gamma)
            w = powed_conv_w / tf.reduce_sum(powed_conv_w)

            if is_read:
                # 3.1 Reading
                read = matmul(tf.transpose(M_prev), w)
                return w, read
            else:
                # 3.2 Writing
                erase = tf.sigmoid(Linear(last_output, self.mem_dim, name='erase_%s' % idx))
                add = tf.tanh(Linear(last_output, self.mem_dim, name='add_%s' % idx))
                return w, add, erase


    def initial_state(self, dummy_value=0.0):
        self.depth = 0
        self.states = []
        with tf.variable_scope("init_cell"):
            # always zero
            dummy = tf.Variable(tf.constant(dummy_value, dtype=tf.float32,
                shape=[1]))

            # memory
            # using dummy to mask out the weight in the linear operation,
            # effectively retaining only the trainable biases
            M_init_linear = tf.tanh(Linear(dummy, self.mem_size * self.mem_dim,
                                    name='M_init_linear'))
            M_init = tf.reshape(M_init_linear, [self.mem_size, self.mem_dim])

            # read weights
            read_w_list_init = []
            read_list_init = []
            for idx in xrange(self.read_head_size):
                read_w_idx = Linear(dummy, self.mem_size, is_range=True,
                                    squeeze=True, name='read_w_%d' % idx)
                read_w_list_init.append(softmax(read_w_idx))

                read_init_idx = Linear(dummy, self.mem_dim,
                                       squeeze=True, name='read_init_%d' % idx)
                read_list_init.append(tf.tanh(read_init_idx))

            # write weights
            write_w_list_init = []
            for idx in xrange(self.write_head_size):
                write_w_idx = Linear(dummy, self.mem_size, is_range=True,
                                     squeeze=True, name='write_w_%s' % idx)
                write_w_list_init.append(softmax(write_w_idx))

            # controller state
            output_init_list = []
            hidden_init_list = []
            for idx in xrange(self.controller_layer_size):
                output_init_idx = Linear(dummy, self.controller_dim,
                                         squeeze=True, name='output_init_%s' % idx)
                output_init_list.append(tf.tanh(output_init_idx))
                hidden_init_idx = Linear(dummy, self.controller_dim,
                                         squeeze=True, name='hidden_init_%s' % idx)
                hidden_init_list.append(tf.tanh(hidden_init_idx))

            # even the output is backed with variables?
            output = tf.tanh(Linear(dummy, self.output_dim, name='new_output'))

            state = {
                'M': M_init,
                'read_w': read_w_list_init,
                'write_w': write_w_list_init,
                'read': read_list_init,
                'output': output_init_list,
                'hidden': hidden_init_list
            }

            self.depth += 1
            self.states.append(state)

        return output, state

    def get_memory(self, depth=None):
        depth = depth if depth else self.depth
        return self.states[depth - 1]['M']

    def get_read_weights(self, depth=None):
        depth = depth if depth else self.depth
        return self.states[depth - 1]['read_w']

    def get_write_weights(self, depth=None):
        depth = depth if depth else self.depth
        return self.states[depth - 1]['write_w']

    def get_read_vector(self, depth=None):
        depth = depth if depth else self.depth
        return self.states[depth - 1]['read']

    def print_read_max(self, sess):
        read_w_list = sess.run(self.get_read_weights())

        fmt = "%-4d %.4f"
        if self.read_head_size == 1:
            print(fmt % (argmax(read_w_list[0])))
        else:
            for idx in xrange(self.read_head_size):
                print(fmt % np.argmax(read_w_list[idx]))

    def print_write_max(self, sess):
        write_w_list = sess.run(self.get_write_weights())

        fmt = "%-4d %.4f"
        if self.write_head_size == 1:
            print(fmt % (argmax(write_w_list[0])))
        else:
            for idx in xrange(self.write_head_size):
                print(fmt % argmax(write_w_list[idx]))
