import tensorflow as tf
from vgg import vgg_16

def main():
    inputs = tf.placeholder(dtype=tf.float32, shape=(None,224,224,3), name="inputs")
    net, end_points = vgg_16(inputs, is_training=False)
    print(end_points)
    print(end_points['vgg_16/conv4/conv4_3'].name)
    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, './vgg_16.ckpt')
        tf.train.write_graph(sess.graph_def, './', 'vgg_16.pbtxt')


if __name__ == '__main__':
    main()
