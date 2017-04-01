# the core part of our neural turing machine, this time with standard LSTM
# controller and hand-crafted batched functions for memory module
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.python.util import nest
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import nn_ops

from ops import batched_smooth_cosine_similarity,  batched_circular_convolution

class NTMCell(object):
    def __init__(self, output_dim, mem_size=128, mem_dim=20, shift_range=1,
            controller_hidden_size=100, controller_num_layers=10,
            write_head_size=3, read_head_size=3, write_first=False):
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
        self.mem_size = mem_size
        self.mem_dim = mem_dim
        self.controller_hidden_size = controller_hidden_size
        self.controller_num_layers = controller_num_layers
        self.write_head_size = write_head_size
        self.read_head_size = read_head_size
        self.shift_range = shift_range
        self.output_dim = output_dim
        self.write_first = write_first

        self.controller = tf.contrib.rnn.MultiRNNCell(
                [tf.contrib.rnn.BasicLSTMCell(
                    controller_hidden_size, forget_bias=0.0,
                    state_is_tuple=False)
                    for x in xrange(controller_num_layers)],
                state_is_tuple=False)


    def __call__(self, inputs, prev_state, M_prev=None, w_prev=None,
            read_prev=None, controller_state=None, scope=None):
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
        with tf.variable_scope(scope or 'ntm-cell'):

            """
            memory-related state
            """
            #memory value of previous timestep
            #[batch, mem_size, mem_dim]
            if prev_state is not None:
                M_prev = prev_state['M']
                #M_prev = tf.Print(M_prev, [M_prev], message="M_prev: ")
                w_prev = prev_state['w']
                #w_prev = tf.Print(w_prev, [w_prev], message="w_prev: ")
                read_prev = prev_state['read'] #read value history
                #read_prev = tf.Print(read_prev, [read_prev], message="read_prev: ")
                #last output. It's a list because controller is an LSTM with multiple cells
                """
                controller state
                """
                controller_state = prev_state['controller_state']

            #shape of controller_output: [batch, controller_hidden_dim]
            #shape of inputs: [batch, num_channels*num_features]
            #shape of target: [batch, num_features]
            #shape of read_prev: [batch, num_read_heads, mem_dim]
            read_prev = tf.reshape(read_prev,
                    [-1, self.read_head_size*self.mem_dim])
            controller_output, controller_state = self.controller(
                    tf.concat([inputs, read_prev], 1),
                    controller_state, scope="lstm-controller")

            # build a memory
            # the memory module simply morph the controller output

            """addressing"""
            # calculate the sizes of parameters
            with tf.variable_scope("addressing"):
                num_heads = self.read_head_size+self.write_head_size

                k_size = self.mem_dim * num_heads
                beta_size = 1 * num_heads
                g_size = 1 * num_heads
                sw_size = (self.shift_range * 2 + 1) * num_heads
                gamma_size = 1 * num_heads
                erase_size = self.mem_dim * self.write_head_size
                add_size = self.mem_dim * self.write_head_size
                # convert the output into the memory_controls matrix ready to be
                # split into control weights. [batch, memory_control_size]
                memory_controls = _linear(controller_output,
                    k_size+beta_size+g_size+sw_size+gamma_size+erase_size+add_size,
                    True, scope="unpack_mem_params")

                k, beta, g, sw, gamma, erase, add = array_ops.split(memory_controls,
                        [k_size, beta_size, g_size, sw_size, gamma_size, erase_size,
                            add_size], axis=1)

                #k is now in [batch, mem_dim * num_heads]
                k = tf.tanh(tf.reshape(k, [-1, num_heads, self.mem_dim]), name='k')
                #k = tf.Print(k, [k, M_prev], message="k and M_prev: ")
                #cos similarity [batch, num_heads, mem_size]
                similarity = batched_smooth_cosine_similarity(M_prev, k)
                #similarity = tf.Print(similarity, [similarity], message="similarity: ")
                #focus by content, result [batch, num_heads, mem_size]
                #beta is [batch, num_heads]
                beta = tf.expand_dims(tf.nn.softplus(beta, name="beta"), -1)
                #beta = tf.Print(beta, [beta], message="beta: ")
                w_content_focused = tf.nn.softmax(tf.multiply(similarity, beta),
                        name="w_content_focused")
                #tf.summary.image('w_cf_step',
                #        tf.reshape(w_content_focused,
                #            [-1, self.mem_size, (self.read_head_size+self.write_head_size), 1]),
                #        max_outputs=32)
                #w_content_focused = tf.Print(w_content_focused,         [w_content_focused], message="w_cf: ")
                #import pdb; pdb.set_trace()
                #g [batch, num_heads]
                g = tf.expand_dims(tf.sigmoid(g, name='g'), -1)
                #w_prev [batch, num_heads, mem_size]
                w_gated = tf.add_n([
                    tf.multiply(w_content_focused, g),
                    tf.multiply(w_prev, (1.0 - g)),
                ], name="w_gated")
                #w_gated = tf.Print(w_gated, [w_gated], message="w_gated: ")
                #convolution shift
                #sw is now in [batch, num_heads * shift_space]
                #afterwards, sw is in [batch, num_heads, shift_space]
                sw = tf.nn.softmax(tf.reshape(sw, [-1, num_heads, self.shift_range * 2 + 1]), name="shift_weight")
                #sw = tf.Print(sw, [sw], message="shift weight: ")

                #[batch, num_heads, mem_size]
                w_conv = batched_circular_convolution(w_gated, sw, name="w_conv")
                #w_conv = tf.Print(w_conv, [w_conv], message="w_conv: ")

                #sharpening
                gamma = tf.expand_dims(tf.add(tf.nn.softplus(gamma),
                    tf.constant(1.0), name="gamma"),-1)

                #gamma = tf.Print(gamma, [gamma], message="gamma: ")
                powed_w_conv = tf.pow(w_conv, gamma, name="powed_w_conv")
                #powed_w_conv = tf.Print(powed_w_conv, [powed_w_conv, tf.reduce_sum(powed_w_conv)], message="powed_w_conv: ")
                w = tf.div(powed_w_conv, tf.reduce_sum(powed_w_conv, axis=2,
                        keep_dims=True) + 1e-3, name='w')
                #w = tf.Print(w, [w], message="w: ")

                #split the read and write head weights
                #w is [batch, num_heads, mem_size]
                w_read = tf.slice(w, [0,0,0],
                        [-1,self.read_head_size,-1], name="w_read")
                w_write = tf.slice(w, [0,self.read_head_size,0], [-1,-1,-1],
                        name="w_write")

                #memory value of previous timestep M_prev is [batch, mem_size, mem_dim]
                #the read result
                #NOTE: moved to the end
                #read = tf.matmul(w_read, M_prev, name="read")

                #now the writing
                #erase [batch, write_head_size, mem_dim]
                erase = tf.sigmoid(tf.reshape(erase,
                    [-1,self.write_head_size,self.mem_dim]), name="erase")
                add = tf.tanh(tf.reshape(add,
                    [-1,self.write_head_size,self.mem_dim]), name="add")
                #w_write [batch, write_head_size, mem_size]
                #M_erase should be [batch, mem_size, mem_dim]
                #calculate M_erase by outer product of w_write and erase
                #after matmul, the dimension becomes [batch, write_head,
                #mem_size, mem_dim]
                M_erase = \
                tf.reduce_prod(1.0 - tf.matmul(tf.expand_dims(w_write,3),
                    tf.expand_dims(erase,2)), axis=1, name="M_erase")
                #calculate M_write by outer product of w_write and add
                M_write = \
                tf.reduce_sum(tf.matmul(tf.expand_dims(w_write,3),
                    tf.expand_dims(add,2)), axis=1, name="M_write")

                M = M_prev * M_erase + M_write

                if self.write_first:
                    read = tf.matmul(w_read, M, name="read")
                else:
                    read = tf.matmul(w_read, M_prev, name="read")

            """ get the real output """
            # by extracting the output tensors from the controller_output, and
            # applying a matrix to change the dimension to target
            ntm_output_logit = _linear(controller_output, self.output_dim, True, scope="unpack_output")
            ntm_output= tf.nn.softmax(ntm_output_logit, name="ntm_output")

            state = {
                'M': M,
                'w': w,
                'read': read,
                'controller_state': controller_state,
            }

            debug = {
                    'k': k,
                    'gamma': gamma,
                    'add': add,
                    'erase': erase,
                    'bega': beta,
                    'g': g,
                    'sw': sw,
                    'similarity': similarity,
                    'w_content_focused': w_content_focused,
                    'w_gated': w_gated,
                    'w_conv': w_conv,
                    'w_conv_powed': powed_w_conv,
                    'w': w,
                    'w_read': w_read,
                    'w_write': w_write,
                    'M': M,
                    'M_prev': M_prev,
                    'M_write': M_write,
                    'M_erase': M_erase,
                    }

        return (ntm_output, ntm_output_logit, state, debug, M, w, read,
                controller_state)

    def state_placeholder(self, batch_size):
        with tf.variable_scope("init_state_ph"):
            M = tf.placeholder(
                    dtype=tf.float32,
                    shape=[batch_size, self.mem_size, self.mem_dim],
                    name="M")
            w = tf.placeholder(
                    dtype=tf.float32,
                    shape=[batch_size,
                        self.read_head_size+self.write_head_size,
                        self.mem_size],
                    name="w")
            read = tf.placeholder(
                    dtype=tf.float32,
                    shape=[batch_size, self.read_head_size, self.mem_dim],
                    name="read")
            controller_state = tf.placeholder(
                    dtype=tf.float32,
                    shape=self.controller.zero_state(batch_size,
                        tf.float32).get_shape(),
                    name="controller_state")
        state = {
                'M': M,
                'w': w,
                'read': read,
                'controller_state': controller_state,
                }
        return state

    def zero_state(self, batch_size, initializer=None):
        """
        zero state should contain:
            1. initial meory value [batch, mem_size, mem_dim]
            2. initial w value [batch, num_heads, mem_size]
            3. initial read value [batch, read_head_size, mem_dim]
            4. initial controller state
        """
        with tf.variable_scope("init_state".format(batch_size)):
            M = tf.tanh(tf.get_variable("M",
                    [self.mem_size, self.mem_dim],
                    dtype=tf.float32, initializer=initializer))
            M = tf.stack([M]*batch_size, 0)
            w = tf.sigmoid(tf.get_variable("w",
                    [self.read_head_size+self.write_head_size,
                        self.mem_size], dtype=tf.float32,
                    initializer=initializer))
            w = tf.stack([w]*batch_size, 0)
            read = tf.tanh(tf.get_variable("read",
                    [self.read_head_size, self.mem_dim],
                    dtype=tf.float32,
                    initializer=initializer))
            read = tf.stack([read]*batch_size, 0)

            controller_state = self.controller.zero_state(batch_size, tf.float32)
        state = {
                'M': M,
                'w': w,
                'read': read,
                'controller_state': controller_state,
                }
        return state

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
      res = math_ops.matmul(array_ops.concat(args, 1), weights)
    if not bias:
      return res
    with vs.variable_scope(outer_scope) as inner_scope:
      inner_scope.set_partitioner(None)
      biases = vs.get_variable(
          "biases", [output_size],
          dtype=dtype,
          initializer=init_ops.constant_initializer(bias_start, dtype=dtype))
  return nn_ops.bias_add(res, biases)
