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

    def zero_state(self, batch_size):
        """
        zero state should contain:
            1. initial meory value [batch, mem_size, mem_dim]
            2. initial w value [batch, num_heads, mem_size]
            3. initial read value [batch, num_read_heads, mem_dim]
            4. initial controller state
        """
        with tf.variable_scope("init_state"):
            M = tf.zeros(
                    [batch_size, self.mem_size, self.mem_dim],
                    name="M")
            w = tf.zeros(
                    [batch_size,
                        self.read_head_size+self.write_head_size,
                        self.mem_size],
                    name="w")
            read = tf.zeros(
                    [batch, self.read_head_size, self.mem_dim],
                    name="read")
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
