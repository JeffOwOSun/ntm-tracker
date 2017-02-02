# the core part of our neural turing machine, this time with standard LSTM
# controller and hand-crafted batched functions for memory module
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import array_ops

from utils import argmax, matmul
from ops import linear, Linear, softmax, outer_product,\
    batched_smooth_cosine_similarity, scalar_mul, circular_convolution

class NTMCell(object):
    def __init__(self, num_features, mem_size=128, mem_dim=20, shift_range=1,
            controller_hidden_size=100, controller_num_layers=10,
            write_head_size=3, read_head_size=3):
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
        self.num_features = num_features
        self.mem_size = mem_size
        self.mem_dim = mem_dim
        self.controller_hidden_size = controller_hidden_size
        self.controller_num_layers = controller_num_layers
        self.write_head_size = write_head_size
        self.read_head_size = read_head_size
        self.shift_range = shift_range

        controller_cell = tf.contrib.rnn.BasicLSTMCell(
                controller_hidden_size, forget_bias=0.0,
                state_is_tuple=True)
        self.controller = tf.contrib.rnn.MultiRNNCell(
                [controller_cell] * controller_num_layers,
                state_is_tuple=True)


    def __call__(self, inputs, target, state, scope=None):
        """
        Args:
            inputs: Input should be extracted features of VGG network.
            For current stage, we only look at invariable scale objects in video
            sequence. That means we only need one layer feature map as input.
            The input will be serialized into a 1D-vector, [num_features x channels]
            The inputs is 2D vector, [steps, num_features x channels]

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
        self.output_dim = target.get_shape().as_list()[0]
        with tf.variable_scope(scope or 'ntm-cell'):

            """
            memory-related state
            """
            #memory value of previous timestep
            #[batch, mem_size, mem_dim]
            M_prev = state['M']
            w_prev = state['w']
            read_prev = state['read'] #read value history
            #last output. It's a list because controller is an LSTM with multiple cells
            """
            controller state
            """
            controller_state = state['controller_state']

            #shape of controller_output: [batch, controller_hidden_dim]
            controller_output, controller_state = self.controller(
                    tf.concat_v2([inputs, target]+read_list_prev, 1), controller_state)
            #TODO: next line is wrong, but I forgot what use controller_hidden has
            #controller_hidden, _ = controller_state

            # build a memory
            # the memory module simply morph the controller output
            #M, read_w_list, write_w_list, read_list = self.build_memory(M_prev,
            #        read_w_list_prev, write_w_list_prev, controller_output)

            """addressing"""
            # calculate the sizes of parameters
            num_heads = self.num_read_heads+self.num_write_heads

            k_size = self.mem_dim * num_heads
            beta_size = 1 * num_heads
            g_size = 1 * num_heads
            sw_size = (self.shift_range * 2 + 1) * num_heads
            gamma_size = 1 * num_heads
            erase_size = self.mem_dim * self.num_write_heads
            add_size = self.mem_dim * self.num_write_heads
            # convert the output into the memory_controls matrix ready to be
            # split into control weights. [batch, memory_control_size]
            memory_controls = _linear(controller_output,
                k_size+beta_size+g_size+sw_size+gamma_size+erase_size+add_size,
                True)

            k, beta, g, sw, gamma, erase, add = array_ops.split(memory_controls,
                    [k_size, beta_size, g_size, sw_size, gamma_size, erase_size,
                        add_size], axis=1)

            #k is now in [batch, mem_dim * num_heads]
            k = tf.tanh(tf.reshape(k, [-1, num_heads, self.mem_dim]), name='k')
            #cos similarity [batch, num_heads, mem_size]
            similarity = batched_smooth_cosine_similarity(M_prev, k)
            #focus by content, result [batch, num_heads, mem_size]
            #beta is [batch, num_heads]
            beta = tf.nn.softplus(beta)
            w_content_focused = tf.nn.softmax(tf.multiply(similarity, beta), dim=2)
            #g [batch, num_heads]
            g = tf.sigmoid(g, name='g')
            #w_prev [batch, num_heads, mem_size]
            w_gated = tf.add_n([
                tf.multiply(w_content_focused, g),
                tf.multiply(w_prev, (tf.constant(1.0) - g)),
            ])
            #convolution shift
            #sw is now in [batch, num_heads * shift_space]
            #afterwards, sw is in [batch, num_heads, shift_space]
            sw = tf.softmax(tf.reshape(sw, [-1, num_heads, self.shift_range * 2
                + 1]), name="shift_weight")

            #[batch, num_heads, mem_size]
            w_conv = batched_circular_convolution(w_gated, sw)

            #sharpening
            gamma = tf.add(tf.nn.softplus(gamma), tf.constant(1.0))
            powed_w_conv = tf.pow(w_conv, gamma)
            w = powed_w_conv / tf.reduce_sum(powed_w_conv, axis=2)

            #split the read and write head weights
            #w is [batch, num_heads, mem_size]
            w_read = tf.slice(w, [0,0,0], [-1,self.num_read_heads,-1])
            w_write = tf.slice(w, [0,self.num_write_heads,0], [-1,-1,-1])

            #memory value of previous timestep M_prev is [batch, mem_size, mem_dim]
            #the read result
            read = tf.batch_matmul(w_read, M_prev)

            #now the writing
            #erase [batch, num_write_heads, mem_dim]
            erase = 1.0 - tf.sigmoid(tf.reshape(erase,
                [-1,self.num_write_heads,self.mem_dim]), name="erase")
            add = tf.tanh(tf.reshape(add,
                [-1,self.num_write_heads,self.mem_dim]), name="add")
            #w_write [batch, num_write_heads, mem_size]
            #M_erase should be [batch, mem_size, mem_dim]
            #calculate M_erase by outer product of w_write and erase
            M_erase = \
            tf.reduce_prod(tf.batch_matmul(tf.expand_dims(w_write,3),
                tf.expand_dims(erase,2)), axis=1)
            #calculate M_write by outer product of w_write and add
            M_write = \
            tf.reduce_sum(tf.batch_matmul(tf.expand_dims(w_write,3),
                tf.expand_dims(add,2)), axis=1)

            M = M_prev * M_erase + M_write

            """ get the real output """
            # by extracting the output tensors from the controller_output, and
            # applying a matrix to change the dimension to target
            ntm_output_logit = _linear(controller_output, self.output_dim, True)
            ntm_output= tf.nn.softmax(ntm_output_logit)

            state = {
                'M': M,
                'w': w,
                'read': read,
                'controller_state': controller_state,
            }

        return ntm_output, ntm_output_logit, state

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

    def build_controller(self, input_, target, read_list_prev,
            output_list_prev, hidden_list_prev):
        #build the controller
        input_ = tf.reshape(input_, [-1])
        output_list = []
        hidden_list = []
        with tf.variable_scope("controller"):
            # for every layer of the lstm
            for layer_idx in xrange(self.controller_layer_size):
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
            dummy = tf.Variable(tf.constant([[dummy_value]], dtype=tf.float32))

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

    def new_output(self, output):
        """Logistic sigmoid output layers."""

        with tf.variable_scope('output'):
            logit = Linear(output, self.output_dim, name='output')
            return tf.nn.softmax(logit), logit

def _linear(args, output_size, bias, bias_start=0.0, scope=None):
  """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.

  Args:
    args: a 2D Tensor or a list of 2D, batch x n, Tensors.
    output_size: int, second dimension of W[i].
    bias: boolean, whether to add a bias term or not.
    bias_start: starting value to initialize the bias; 0 by default.
    scope: (optional) Variable scope to create parameters in.

  Returns:
    A 2D Tensor with shape [batch x output_size] equal to
    sum_i(args[i] * W[i]), where W[i]s are newly created matrices.

  Raises:
    ValueError: if some of the arguments has unspecified or wrong shape.
  """
  if args is None or (nest.is_sequence(args) and not args):
    raise ValueError("`args` must be specified")
  if not nest.is_sequence(args):
    args = [args]

  # Calculate the total size of arguments on dimension 1.
  total_arg_size = 0
  shapes = [a.get_shape() for a in args]
  for shape in shapes:
    if shape.ndims != 2:
      raise ValueError("linear is expecting 2D arguments: %s" % shapes)
    if shape[1].value is None:
      raise ValueError("linear expects shape[1] to be provided for shape %s, "
                       "but saw %s" % (shape, shape[1]))
    else:
      total_arg_size += shape[1].value

  dtype = [a.dtype for a in args][0]

  # Now the computation.
  scope = vs.get_variable_scope()
  with vs.variable_scope(scope) as outer_scope:
    weights = vs.get_variable(
        "weights", [total_arg_size, output_size], dtype=dtype)
    if len(args) == 1:
      res = math_ops.matmul(args[0], weights)
    else:
      res = math_ops.matmul(array_ops.concat_v2(args, 1), weights)
    if not bias:
      return res
    with vs.variable_scope(outer_scope) as inner_scope:
      inner_scope.set_partitioner(None)
      biases = vs.get_variable(
          "biases", [output_size],
          dtype=dtype,
          initializer=init_ops.constant_initializer(bias_start, dtype=dtype))
  return nn_ops.bias_add(res, biases)
