import math
import numpy as np
import tensorflow as tf

from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope as vs

from utils import *

def linear(args, output_size, bias, bias_start=0.0, scope=None):
    """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.

    Args:
        args: a 2D Tensor or a list of 2D, batch x n, Tensors.
        output_size: int, second dimension of W[i].
        bias: boolean, whether to add a bias term or not.
        bias_start: starting value to initialize the bias; 0 by default.
        scope: VariableScope for the created subgraph; defaults to "Linear".

    Returns:
        A 2D Tensor with shape [batch x output_size] equal to
        sum_i(args[i] * W[i]), where W[i]s are newly created matrices.

    Raises:
        ValueError: if some of the arguments has unspecified or wrong shape.
    """
    if not isinstance(args, (list, tuple)):
        args = [args]

    # Calculate the total size of arguments on dimension 1.
    total_arg_size = 0
    # collect the shape information of all the arguments
    shapes = []
    for a in args:
        try:
            shapes.append(a.get_shape().as_list())
        except Exception as e:
            shapes.append(a.shape)

    is_vector = False
    for idx, shape in enumerate(shapes):
        if len(shape) != 2:
            # Reshape all the arguments that are not 2D into 2D matrix.
            # If one argument is 2D, its dimentsions are batch x n_feature.
            # When one argument is 1D, assume its dimention as n_feature.
            # The reshape operation simply put it into 1 x n_feature.
            is_vector = True
            args[idx] = tf.reshape(args[idx], [1, -1])
            total_arg_size += shape[0]
        else:
            total_arg_size += shape[1]

    # Now the computation.
    with vs.variable_scope(scope or "Linear"):
        # Here the parameter is not passed in.
        # Instead, a variable named Matrix is created/retrieved
        # The scope passed in handles the disambiguation of the variable
        matrix = vs.get_variable("Matrix", [total_arg_size, output_size])
        if len(args) == 1:
            res = math_ops.matmul(args[0], matrix)
        else:
            res = math_ops.matmul(array_ops.concat_v2(args, 1), matrix)
        if not bias:
            return res
        bias_term = vs.get_variable(
            "Bias", [output_size],
            initializer=init_ops.constant_initializer(bias_start))

    # reshape the result back to vector if input is vector
    if is_vector:
        return tf.reshape(res + bias_term, [-1])
    else:
        return res + bias_term

def Linear(input_, output_size, stddev=0.5,
           is_range=False, squeeze=False,
           name=None, reuse=None):
    """
    Applies a matmult and a bias add to the incoming data.
    The necessary variables are created within this function with
    get_variable().

    Args:
        input: a 2-D or 1-D data (`Tensor` or `ndarray`)
        output_size: the size of output matrix or vector
    """
    with tf.variable_scope("Linear", reuse=reuse):
        if type(input_) == np.ndarray:
            shape = input_.shape
        else:
            shape = input_.get_shape().as_list()

        is_vector = False
        if len(shape) == 1:
            is_vector = True
            input_ = tf.reshape(input_, [1, -1])
            input_size = shape[0]
        elif len(shape) == 2:
            input_size = shape[1]
        else:
            raise ValueError("Linear expects shape[1] of inputuments: %s" % str(shape))

        w_name = "%s_w" % name if name else None
        b_name = "%s_b" % name if name else None

        w = tf.get_variable(w_name, [input_size, output_size], tf.float32,
                            tf.random_normal_initializer(stddev=stddev))
        mul = tf.matmul(input_, w)

        if is_range:
            def identity_initializer(tensor):
                def _initializer(shape, dtype=tf.float32, partition_info=None):
                    return tf.identity(tensor)
                return _initializer

            range_ = tf.reverse_v2(tf.range(1, output_size+1, 1, dtype=tf.float32), [0])
            b = tf.get_variable(b_name, [output_size], tf.float32,
                                identity_initializer(range_))
        else:
            b = tf.get_variable(b_name, [output_size], tf.float32,
                                tf.random_normal_initializer(stddev=stddev))

        if squeeze:
            output = tf.squeeze(tf.nn.bias_add(mul, b))
        else:
            output = tf.nn.bias_add(mul, b)

        if is_vector:
            return tf.reshape(output, [-1])
        else:
            return output

def batched_smooth_cosine_similarity(memory, keys):
    """batched version of smooth cosine similarity

    Args:
        memory: [batch, mem_size, mem_dim]
        keys: [batch, num_heads, mem_dim]

    return:
        similarity: [batch, num_heads, mem_size]
    """
    #memory_norm [batch, mem_size, 1]
    memory_norm = tf.sqrt(tf.reduce_sum(tf.pow(memory, 2),2,keep_dims=True))
    #keys_norm [batch, num_heads, 1]
    keys_norm = tf.sqrt(tf.reduce_sum(tf.pow(keys, 2),2,keep_dims=True))
    #dot_product [batch, num_heads, mem_size]
    dot_product = tf.batch_matmul(keys, memory, adj_y=True)
    norm_product = tf.batch_matmul(keys_norm, memory_norm, adj_y=True)
    return tf.div(dot_product, norm_product)


def smooth_cosine_similarity(m, v):
    """Computes smooth cosine similarity.

    Args:
        m: a 2-D `Tensor` (matrix)
        v: a 1-D `Tensor` (vector)
    """
    shape_x = m.get_shape().as_list()
    shape_y = v.get_shape().as_list()
    if shape_x[1] != shape_y[0]:
        raise ValueError("Smooth cosine similarity is expecting same dimemsnion")

    m_norm = tf.sqrt(tf.reduce_sum(tf.pow(m, 2),1))
    v_norm = tf.sqrt(tf.reduce_sum(tf.pow(v, 2)))
    m_dot_v = tf.matmul(m, tf.reshape(v, [-1, 1]))

    similarity = tf.div(tf.reshape(m_dot_v, [-1]), m_norm * v_norm + 1e-3)
    return similarity

def batched_circular_convolution(tensor, kernel):
    """
    batched version of circular convolution
    the first two dimensions of the input are just batch sizes, and on slice
    level it's just a 1D vector convoluted with a 1D kernel

    Args:
        tensor: [batch, num_heads, mem_size]
        kernel: [batch, num_heads, shift_space]
    return:
        convoluted_tensor: [batch, num_heads, mem_size]
    """
    return

def circular_shift(tensor, shift):
    """
    Shift the given tensor by steps on the LAST axis.
    A positive step means that X[..., i] == X'[..., i+1]

    Args:
        tensor: [..., dim]
        shift: an integer, smaller than dim
    return:
        a tensor of the same shape as tensor
    """
    shape = tensor.get_shape().as_list()

    #NOTE: assuming a positive shift
    splitting_point = shape[-1]-shift if shift > 0 else 0-shift

    start = [0]*len(shape)
    start[-1] = splitting_point
    size = [-1]*len(shape)
    left_slice = tf.slice(tensor, start, size)

    start = [0]*len(shape)
    size = [-1]*len(shape)
    size[-1] = splitting_point
    right_slice = tf.slice(tensor, start, size)
    return tf.concat_v2([left_slice, right_slice], axis=-1)

def circular_convolution(v, k):
    """Computes circular convolution.

    Args:
        v: a 1-D `Tensor` (vector)
        k: a 1-D `Tensor` (kernel)
    """
    size = int(v.get_shape()[0])
    kernel_size = int(k.get_shape()[0])
    kernel_shift = int(math.floor(kernel_size/2.0))

    def loop(idx):
        if idx < 0: return size + idx
        if idx >= size : return idx - size
        else: return idx

    kernels = []
    for i in xrange(size):
        indices = [loop(i+j) for j in xrange(kernel_shift, -kernel_shift-1, -1)]
        v_ = tf.gather(v, indices)
        kernels.append(tf.reduce_sum(v_ * k, 0))

    # # code with double loop
    # for i in xrange(size):
    #     for j in xrange(kernel_size):
    #         idx = i + kernel_shift - j + 1
    #         if idx < 0: idx = idx + size
    #         if idx >= size: idx = idx - size
    #         w = tf.gather(v, int(idx)) * tf.gather(kernel, j)
    #         output = tf.scatter_add(output, [i], tf.reshape(w, [1, -1]))

    return tf.dynamic_stitch([i for i in xrange(size)], kernels)

def outer_product(*inputs):
    """Computes outer product.

    Args:
        inputs: a list of 1-D `Tensor` (vector)
    """
    inputs = list(inputs)
    order = len(inputs)

    for idx, input_ in enumerate(inputs):
        if len(input_.get_shape()) == 1:
            inputs[idx] = tf.reshape(input_, [-1, 1] if idx % 2 == 0 else [1, -1])

    if order == 2:
        output = tf.mul(inputs[0], inputs[1])
    elif order == 3:
        size = []
        idx = 1
        for i in xrange(order):
            size.append(inputs[i].get_shape()[0])
        output = tf.zeros(size)

        u, v, w = inputs[0], inputs[1], inputs[2]
        uv = tf.mul(inputs[0], inputs[1])
        for i in xrange(self.size[-1]):
            output = tf.scatter_add(output, [0,0,i], uv)

    return output

def scalar_mul(x, beta, name=None):
    return x * beta

def scalar_div(x, beta, name=None):
    return x / beta
