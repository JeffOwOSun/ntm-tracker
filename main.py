from __future__ import absolute_import

import tensorflow as tf
import numpy as np
import pickle
from datetime import datetime
import os

from vgg import vgg_16

from ntm import NTMTracker

import random

flags = tf.app.flags
#flags.DEFINE_string("task", "copy", "Task to run [copy, recall]")
#flags.DEFINE_integer("epoch", 100000, "Epoch to train [100000]")
#flags.DEFINE_integer("input_dim", 10, "Dimension of input [10]")
#flags.DEFINE_integer("output_dim", 10, "Dimension of output [10]")
#flags.DEFINE_integer("min_length", 1, "Minimum length of input sequence [1]")
#flags.DEFINE_integer("max_length", 10, "Maximum length of output sequence [10]")
#flags.DEFINE_integer("controller_layer_size", 1, "The size of LSTM controller [1]")
#flags.DEFINE_integer("controller_dim", 100, "Dimension of LSTM controller [100]")
#flags.DEFINE_integer("write_head_size", 1, "The number of write head [1]")
#flags.DEFINE_integer("read_head_size", 1, "The number of read head [1]")
#flags.DEFINE_integer("test_max_length", 120, "Maximum length of output sequence [120]")
#flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
#flags.DEFINE_boolean("is_train", False, "True for training, False for testing [False]")
#flags.DEFINE_boolean("continue_train", None, "True to continue training from saved checkpoint. False for restarting. None for automatic [None]")
flags.DEFINE_integer("num_epochs", 1, "number of epochs to train")
flags.DEFINE_string("vgg_model_frozen", "./vgg_16_frozen.pb", "The pb file of the frozen vgg_16 network")
flags.DEFINE_boolean("test_read_imgs", False, "test read imgs module")
flags.DEFINE_boolean("lstm_only", False, "use build-in lstm only")
flags.DEFINE_string("log_dir", "/tmp/ntm-tracker", "The log dir")
flags.DEFINE_integer("sequence_length", 20, "The length of fixed sequences")
flags.DEFINE_string("feature_layer", "vgg_16/conv4/conv4_3/Relu:0", "The layer of feature to be put into NTM as input")
flags.DEFINE_integer("max_gradient_norm", 5, "for gradient clipping normalization")
flags.DEFINE_float("learning_rate", 1e-4, "learning rate")
flags.DEFINE_float("momentum", 0.9, "learning rate")
flags.DEFINE_float("decay", 0.95, "learning rate")
flags.DEFINE_integer("hidden_size", 100, "number of LSTM cells")
flags.DEFINE_integer("num_layers", 100, "number of LSTM cells")
flags.DEFINE_string("tag", "", "tag for the log record")

FLAGS = flags.FLAGS

random.seed(42)

def create_ntm(inputs, target_first_frame, sequence_length=20):
    ntm = NTMTracker(max_sequence_length=sequence_length)
    outputs, output_logits, states = ntm(inputs, target_first_frame)
    return outputs, output_logits, states


def create_vgg(inputs, feature_layer):
    net, end_points = vgg_16(inputs)
    print(end_points.keys())
    return end_points[feature_layer]

def get_batch(index, batch_size, seq_length, seqs):
    seq_batch = seqs[index:index+batch_size]
    index+=batch_size
    frame_names = []
    real_gts = []
    for seq_dir, obj_name, subseq_id, seq_len, seq in seq_batch:
        # only need the first seq_length frames
        seq = seq[:sequence_length]
        # the file names [batch * seq_length]
        frame_names += [x[0] for x in seq]
        # the ground truths [batch, seq_length]
        real_gts.append(np.array([np.reshape(x[-1][0], (-1)) for x in seq]))
    real_gts = np.array(real_gts)
    return frame_names, real_gts, index

def read_imgs(batch_size, seq_length):
    # a fifo queue with 100 capacity
    filename_queue = tf.FIFOQueue(batch_size*seq_length, tf.string)
    # the entrance placeholder to the pipeline
    enqueue_placeholder = tf.placeholder(shape=(batch_size*seq_length), dtype=tf.string)
    # the opration to be run to enqueue the real filenames
    enqueue_op = filename_queue.enqueue_many(enqueue_placeholder)
    # will be called after everything is done
    queue_close_op = filename_queue.close()
    # reader to convert file names to actual data
    reader = tf.WholeFileReader()
    key, value = reader.read(filename_queue)
    # here value represents one instance of image
    my_img = tf.image.decode_jpeg(value)
    my_img = tf.reshape(tf.image.resize_images(my_img, [224, 224]), (224, 224, 3))
    my_img = tf.image.per_image_standardization(my_img)
    # convert the queue-based image stream into a batch
    batch_img = tf.train.batch([my_img],
            batch_size = batch_size*seq_length,
            num_threads = 1)
    batch_img = tf.reshape(batch_img, [batch_size, seq_length, 224, 224, 3])
    return enqueue_placeholder, enqueue_op, queue_close_op, batch_img

def test_read_imgs():
    with tf.Session() as sess:
        test_img_name = '/home/jowos/data/ILSVRC2015/Data/VID/train/a/ILSVRC2015_train_00139005/000379.JPEG'
        enqueue_placeholder, enqueue_op, queue_close_op, batch_img = read_imgs(20)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        feed_dict = {
                enqueue_placeholder:
                20*[test_img_name]
                }
        sess.run(enqueue_op, feed_dict=feed_dict)
        features = create_vgg(batch_img, FLAGS.feature_layer)
        saver = tf.train.Saver()
        saver.restore(sess, "./vgg_16.ckpt")
        output = sess.run(features)
        print(output.shape)

        sess.run(queue_close_op) #close the queue
        coord.request_stop()
        coord.join(threads)

def train_and_val(train_op, loss, merged, target, gt,
        file_names_placeholder, enqueue_op, q_close_op):
    with tf.Session() as sess:
        print('session started')
        writer = tf.summary.FileWriter(os.path.join(FLAGS.log_dir,
                str(datetime.now())+FLAGS.tag), sess.graph)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        # initialize variables
        sess.run(tf.initialize_all_variables())
        print("start to run the training.")
        """
        1. get the statistics
        2. get the images
        3. extract the features
        4. train the network
        """
        print("loading generated_sequences.pkl...")
        with open('generated_sequences.pkl', 'r') as f:
            generated_sequences = pickle.load(f)
        #shuffle the order
        print("shuffling the sequences...")
        random.shuffle(generated_sequences)
        #filter the short sequences
        print("filtering out too short sequences...")
        generated_sequences = [x for x in generated_sequences if x[-2] >=
                FLAGS.sequence_length]
        print('{} sequences after length filtering'.format(len(generated_sequences)))
        #divide train/test batches
        num_train = (len(generated_sequences)/10*9)/FLAGS.batch_size*FLAGS.batch_size
        num_test = (len(generated_sequences)/10)/FLAGS.batch_size*FLAGS.batch_size
        test_seqs = generated_sequences[:num_test]
        train_seqs = generated_sequences[-num_train:]
        print('{} train seqs, {} test seqs'.format(
            len(train_seqs), len(test_seqs)))
        step = 0
        num_epochs = FLAGS.num_epochs
        for epoch in xrange(num_epochs):
            print("training epoch {}".format(epoch))
            random.shuffle(train_seqs)
            print("shuffled training seqs")
            #train
            index = 0
            while index < len(train_seqs):
                # this batch
                frame_names, real_gts, index = get_batch(index,
                        FLAGS.batch_size, FLAGS.seq_length, train_seqs)
                feed_dict = {file_names_placeholder:
                            frame_names}
                #print(feed_dict)
                sess.run(enqueue_op, feed_dict=feed_dict)
                # extract the ground truths
                # finally it will be a 2D array

                real_loss, _, summary = sess.run((loss, train_op, merged),
                    feed_dict = {
                        target: np.reshape(real_gts[0], [1, -1]),
                        gt: real_gts,
                    })
                writer.add_summary(summary, step)
                if step % 10 == 0:
                    print("{}: training loss {}".format(step, real_loss))
                step += 1

        step = 0
        accumu_loss = 0
        index = 0
        while index < len(test_seqs):
            frame_names, real_gts, index = get_batch(index,
                    FLAGS.batch_size, FLAGS.seq_length, test_seqs)
            feed_dict = {file_names_placeholder:
                        frame_names}
            sess.run(enqueue_op, feed_dict=feed_dict)
            # extract the ground truths
            real_loss = sess.run(loss, feed_dict = {
                    target: np.reshape(real_gts[0], [1, -1]),
                    gt: real_gts,
                })
            accumu_loss += real_loss
            step += 1
        print("average testing loss {}".format(accumu_loss / float(step)))
        saver = tf.train.Saver()
        save_path = saver.save(sess, "model.ckpt")
        print("model saved to {}".format(save_path))

        sess.run(q_close_op) #close the queue
        coord.request_stop()
        coord.join(threads)

def lstm_only():
    print("creating model")
    """
    create the model
    """
    """get the inputs"""
    file_names_placeholder, enqueue_op, q_close_op, batch_img = read_imgs(FLAGS.sequence_length)
    vgg_graph_def = tf.GraphDef()
    with open(FLAGS.vgg_model_frozen, "rb") as f:
        vgg_graph_def.ParseFromString(f.read())
    features = tf.import_graph_def(vgg_graph_def, input_map={'inputs':
        batch_img}, return_elements=[FLAGS.feature_layer])[0]
    features_dim = features.get_shape().as_list()
    num_features = features_dim[1]*features_dim[2]
    """compress input dimensions"""
    w = tf.get_variable('input_compressor_w',
            shape=(1,1,features_dim[-1],128), dtype=tf.float32,
            initializer=tf.contrib.layers.xavier_initializer())
    features = tf.nn.conv2d(features, w, strides=(1,1,1,1), padding="VALID",
            name="input_compressor")

    """the lstm"""
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(
            FLAGS.hidden_size, forget_bias=0.0, state_is_tuple=True)
    cell = tf.contrib.rnn.MultiRNNCell(
            [lstm_cell] * FLAGS.num_layers, state_is_tuple=True)
    """initial state"""
    initial_state = cell.zero_state(1, tf.float32)

    #TODO: make sure this reshape is working as expected
    """inputs and outputs to the lstm"""
    inputs = tf.reshape(features, shape=[1, FLAGS.sequence_length, -1])
    target_ph = tf.placeholder(tf.float32,
            shape=[1, num_features], name="target")
    dummy_target = tf.constant(0.0, shape=target_ph.get_shape())
    gt_ph = tf.placeholder(tf.float32,
        shape=[FLAGS.sequence_length, num_features], name="ground_truth")
    """actually build the lstm"""
    print("building lstm")
    outputs = []
    state = initial_state
    with tf.variable_scope("ntm-tracker"):
        for time_step in range(FLAGS.sequence_length):
            if time_step > 0:
                tf.get_variable_scope().reuse_variables()
                cell_output, state = cell(
                    tf.concat_v2([inputs[:, time_step, :], dummy_target],1), state)
            else:
                cell_output, state = cell(
                    tf.concat_v2([inputs[:, time_step, :], target_ph],1), state)
            outputs.append(cell_output)
    output = tf.reshape(tf.concat_v2(outputs, 1), [-1, FLAGS.hidden_size])
    """compress the output to our desired dimensions"""
    softmax_w = tf.get_variable(
        "softmax_w", [FLAGS.hidden_size, num_features], dtype=tf.float32)
    softmax_b = tf.get_variable("softmax_b", [num_features], dtype=tf.float32)
    output_logits = tf.matmul(output, softmax_w) + softmax_b
    """loss"""
    loss_op = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(output_logits,
        tf.nn.softmax(gt_ph))) / FLAGS.sequence_length
    tf.summary.scalar('loss', loss_op)
    """training op"""
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(loss_op, tvars),
            FLAGS.max_gradient_norm)
    lr = tf.constant(FLAGS.learning_rate, name="learning_rate")
    optimizer = tf.train.GradientDescentOptimizer(lr)
    train_op = optimizer.apply_gradients(
            zip(grads, tvars),
            global_step = tf.contrib.framework.get_or_create_global_step())
    merged_summary = tf.summary.merge_all()

    train_and_val(train_op, loss_op, merged_summary, target_ph, gt_ph,
            file_names_placeholder, enqueue_op, q_close_op)


def main(_):
    """
    create the graph
    """
    """ 1. the img input preprocessor """
    file_names_placeholder, enqueue_op, q_close_op, batch_img = read_imgs(FLAGS.sequence_length)
    """ 2. the VGG feature extractor """
    vgg_graph_def = tf.GraphDef()
    with open(FLAGS.vgg_model_frozen, "rb") as f:
        vgg_graph_def.ParseFromString(f.read())
    features = tf.import_graph_def(vgg_graph_def, input_map={'inputs': batch_img},
            return_elements=[FLAGS.feature_layer])[0]
    features_dim = features.get_shape().as_list()
    #feature is [20, 28, 28, 512]
    #compress the feature down to 128 channels
    print('building ntm')
    with tf.variable_scope('ntm'):
        w = tf.get_variable('input_compressor_w',
                shape=(1,1,features_dim[-1],128), dtype=tf.float32,
                initializer=tf.contrib.layers.xavier_initializer())
        features = tf.nn.conv2d(features, w, strides=(1,1,1,1), padding="VALID",
                name="input_compressor")
        target = tf.placeholder(tf.float32,
                shape=[features_dim[1]*features_dim[2]], name="target")
        gt = tf.placeholder(tf.float32,
            shape=[FLAGS.sequence_length, target.get_shape().as_list()[0]], name="ground_truth")
    outputs, output_logits, states = create_ntm(features, target,
            FLAGS.sequence_length)
    with tf.variable_scope('ntm'):
        # softmax is applied to gt due to the requirement of
        # softmax_cross_entropy_with_logits
        # NOTE using the softmax cross entroy built-in to tensorflow.
        # Originally this function is designed to take in logits of dimension
        # [batch, num_classes] but here I'm using it as [len_sequence,
        # num_features]
        # NOTE the output loss is a 1D vector of shape [len_sequence]
        # I'll need to decide how to pair this with backprop
        loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(output_logits,
            tf.nn.softmax(gt))) / FLAGS.sequence_length
        tf.summary.scalar('loss', loss)
    # get the gradient
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars),
                FLAGS.max_gradient_norm)
        optimizer = tf.train.RMSPropOptimizer(FLAGS.learning_rate,
                decay=FLAGS.decay, momentum=FLAGS.momentum)
        train_op = optimizer.apply_gradients(
                zip(grads, tvars),
                global_step = tf.contrib.framework.get_or_create_global_step())
        merged = tf.summary.merge_all()
    with tf.Session() as sess:
        print('session started')
        writer = tf.summary.FileWriter(os.path.join(FLAGS.log_dir,
            str(datetime.now())+FLAGS.tag), sess.graph)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        # initialize variables
        sess.run(tf.initialize_all_variables())
        print("start to run the training.")
        """
        1. get the statistics
        2. get the images
        3. extract the features
        4. train the network
        """
        with open('generated_sequences.pkl', 'r') as f:
            generated_sequences = pickle.load(f)
        #shuffle the order
        random.shuffle(generated_sequences)
        #filter the short sequences
        generated_sequences = [x for x in generated_sequences if x[-2] >=
                FLAGS.sequence_length]
        print('{} sequences after length filtering'.format(len(generated_sequences)))
        #divide train/test batches
        test_seqs = generated_sequences[:len(generated_sequences)/10]
        train_seqs = generated_sequences[len(generated_sequences)/10:]
        num_epochs = FLAGS.num_epochs
        step = 0
        for epoch in xrange(num_epochs):
            print("training epoch {}".format(epoch))
            random.shuffle(train_seqs)
            print("shuffled training seqs")
            #train
            for seq_dir, obj_name, subseq_id, seq_len, seq in train_seqs:
                # enqueue the filenames
                seq = seq[:FLAGS.sequence_length]
                feed_dict = {file_names_placeholder:
                            [x[0] for x in seq]}
                #print(feed_dict)
                sess.run(enqueue_op, feed_dict=feed_dict)
                # extract the ground truths
                # finally it will be a 2D array
                real_gts = np.array([np.reshape(x[-1][0], (-1)) for x in seq])

                real_loss, _, summary = sess.run((loss, train_op, merged),
                        feed_dict = {
                            target: real_gts[0],
                            gt: real_gts,
                        })
                writer.add_summary(summary, step)
                if step % 10 == 0:
                    print("{}: training loss {}".format(step, real_loss))
                step += 1

        step = 0
        accumu_loss = 0
        for seq_dir, obj_name, subseq_id, seq_len, seq in test_seqs:
            # enqueue the filenames
            seq = seq[:FLAGS.sequence_length]
            feed_dict = {file_names_placeholder:
                        [x[0] for x in seq]}
            sess.run(enqueue_op, feed_dict=feed_dict)
            # extract the ground truths
            # finally it will be a 2D array
            real_gts = np.array([np.reshape(x[-1][0], (-1)) for x in seq])

            real_loss = sess.run(loss, feed_dict = {
                    target: real_gts[0],
                    gt: real_gts,
                })
            accumu_loss += real_loss
            step += 1
        print("average testing loss {}".format(accumu_loss / float(step)))
        saver = tf.train.Saver()
        save_path = saver.save(sess, "model.ckpt")
        print("model saved to {}".format(save_path))

        sess.run(q_close_op) #close the queue
        coord.request_stop()
        coord.join(threads)

if __name__ == '__main__':
    if (FLAGS.test_read_imgs):
        test_read_imgs()
    elif (FLAGS.lstm_only):
        lstm_only()
    else:
        tf.app.run()
