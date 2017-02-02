import tensorflow as tf

def primary_rnn_cell(input_):
    input_shape = input_.get_shape().as_list()
    with tf.variable_scope('rnn_cell'):
        W = tf.get_variable('matrix_weight',
                shape=[input_shape[-1],input_shape[-1]], dtype=tf.float32)
        b = tf.get_variable('matrix_bias',
                shape=[input_shape[-1]], dtype=tf.float32)

def main():
    with tf.variable_scope('yoo') as scope:
        x = tf.get_variable('x', [1])
    with tf.variable_scope(scope):
        y = tf.get_variable('y', [1])
    with tf.variable_scope(scope, reuse=True):
        z = tf.get_variable('x', [1])
    with tf.variable_scope('yoo') as scope:
        a = tf.get_variable('z', [1])
    with tf.variable_scope('yoo', reuse=True):
        w = tf.get_variable('x', [1])
    print(tuple(x.name for x in (x,y,z,w,a)))

    with tf.variable_scope('yo') as scope:
        print(scope.name)
    with tf.variable_scope('yo') as scope:
        print(scope.name)


if __name__ == '__main__':
    main()
