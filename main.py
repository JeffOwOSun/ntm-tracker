from __future__ import absolute_import

import tensorflow as tf
import numpy as np
import pickle
from datetime import datetime
import os

from vgg import vgg_16

from ntm_tracker_new import NTMTracker

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
flags.DEFINE_integer("batch_size", 16, "size of batch")
flags.DEFINE_string("feature_layer", "vgg_16/conv4/conv4_3/Relu:0", "The layer of feature to be put into NTM as input")
flags.DEFINE_integer("max_gradient_norm", 5, "for gradient clipping normalization")
flags.DEFINE_float("learning_rate", 1e-4, "learning rate")
flags.DEFINE_float("momentum", 0.9, "learning rate")
flags.DEFINE_float("decay", 0.95, "learning rate")
flags.DEFINE_integer("hidden_size", 100, "number of LSTM cells")
flags.DEFINE_integer("num_layers", 10, "number of LSTM cells")
flags.DEFINE_string("tag", "", "tag for the log record")
flags.DEFINE_integer("log_interval", 10, "number of epochs before log")
flags.DEFINE_float("init_scale", 0.05, "initial range for weights")

FLAGS = flags.FLAGS

random.seed(42)

real_log_dir = os.path.join(FLAGS.log_dir, str(datetime.now())+FLAGS.tag)

VGG_MEAN = tf.constant([123.68, 116.78, 103.94], dtype=tf.float32,
        shape=[1,1,3], name="VGG_MEAN")

def create_vgg(inputs, feature_layer):
    net, end_points = vgg_16(inputs)
    print(end_points.keys())
    return end_points[feature_layer]

def get_batch(index, batch_size, seq_length, seqs):
    """
    get a batch of frame names and their ground truths

    seqs: the sequence statistics
    seq_length: the length of subsequence to take
    batch_size: the number of sequences to push into the batch
    """
    seq_batch = seqs[index:index+batch_size]
    index+=batch_size
    frame_names = []
    real_gts = []
    for seq_dir, obj_name, subseq_id, seq_len, seq in seq_batch:
        # only need the first seq_length frames
        seq = seq[:seq_length]
        # the file names [batch * seq_length]
        frame_names += [x[0] for x in seq]
        # the ground truths [batch, seq_length, num_features]
        real_gts.append(np.array([np.reshape(x[-1][0], (-1)) for x in seq]))
    real_gts = np.array(real_gts)
    return frame_names, real_gts, index

def read_imgs(batch_size):
    # a fifo queue with 100 capacity
    filename_queue = tf.FIFOQueue(batch_size, tf.string)
    # the entrance placeholder to the pipeline
    enqueue_placeholder = tf.placeholder(shape=(batch_size), dtype=tf.string)
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
    #my_img = tf.image.per_image_standardization(my_img)
    my_img = my_img - VGG_MEAN
    # convert the queue-based image stream into a batch
    batch_img = tf.train.batch([my_img],
            batch_size = batch_size,
            num_threads = 1)
    tf.summary.image('batch_img', batch_img, max_outputs=batch_size)
    return enqueue_placeholder, enqueue_op, queue_close_op, batch_img

def test_read_imgs():
    #TODO: update this function
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
        file_names_placeholder, enqueue_op, q_close_op, other_ops=[]):
    check_op = tf.add_check_numerics_ops()
    with tf.Session() as sess:
        print('session started')
        writer = tf.summary.FileWriter(real_log_dir, sess.graph)
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
                        FLAGS.batch_size, FLAGS.sequence_length, train_seqs)
                feed_dict = {file_names_placeholder:
                            frame_names}
                #print(feed_dict)
                sess.run(enqueue_op, feed_dict=feed_dict)
                # extract the ground truths
                # finally it will be a 2D array

                ret = sess.run(
                        [loss, train_op, merged]+other_ops,
                        feed_dict = {
                            target: real_gts[:,0,:],
                            gt: real_gts,
                        })
                real_loss, _, summary = ret[:3]
                writer.add_summary(summary, step)
                if step % FLAGS.log_interval == 0:
                    print("{}: training loss {}".format(step, real_loss))
                #import pdb; pdb.set_trace()
                step += 1

        step = 0
        accumu_loss = 0
        index = 0
        while index < len(test_seqs):
            frame_names, real_gts, index = get_batch(index,
                    FLAGS.batch_size, FLAGS.sequence_length, test_seqs)
            feed_dict = {file_names_placeholder:
                        frame_names}
            sess.run(enqueue_op, feed_dict=feed_dict)
            # extract the ground truths
            real_loss = sess.run(loss, feed_dict = {
                    target: real_gts[:,0,:],
                    gt: real_gts,
                })
            accumu_loss += real_loss
            step += 1
        print("average testing loss {}".format(accumu_loss / float(step)))
        saver = tf.train.Saver()
        save_path = saver.save(sess, os.path.join(real_log_dir,
            "model.ckpt"))
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
    file_names_placeholder, enqueue_op, q_close_op, batch_img = read_imgs(FLAGS.batch_size*FLAGS.sequence_length)
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
    initial_state = cell.zero_state(FLAGS.batch_size, tf.float32)

    #TODO: make sure this reshape is working as expected
    """inputs and outputs to the lstm"""
    inputs = tf.reshape(features, shape=[FLAGS.batch_size, FLAGS.sequence_length, -1])
    target_ph = tf.placeholder(tf.float32,
            shape=[FLAGS.batch_size, num_features], name="target")
    dummy_target = tf.constant(0.0, shape=target_ph.get_shape())
    gt_ph = tf.placeholder(tf.float32,
        shape=[FLAGS.batch_size, FLAGS.sequence_length, num_features], name="ground_truth")
    tf.summary.image("ground_truth", tf.reshape(gt_ph,
        [-1,features_dim[1],features_dim[2],1]),
        max_outputs=FLAGS.batch_size*FLAGS.sequence_length)
    """actually build the lstm"""
    print("building lstm")
    outputs = []
    state = initial_state
    with tf.variable_scope("lstm-tracker"):
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
    tf.summary.image("outputs", tf.reshape(tf.nn.softmax(output_logits),
        [-1,features_dim[1],features_dim[2],1]),
        max_outputs=FLAGS.batch_size*FLAGS.sequence_length)
    """loss"""
    loss_op = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(output_logits,
        tf.nn.softmax(gt_ph))) / FLAGS.sequence_length
    tf.summary.scalar('loss', loss_op)
    """training op"""
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(loss_op, tvars),
            FLAGS.max_gradient_norm)
    #lr = tf.constant(FLAGS.learning_rate, name="learning_rate")
    #optimizer = tf.train.GradientDescentOptimizer(lr)
    optimizer = tf.train.RMSPropOptimizer(FLAGS.learning_rate,
            decay=FLAGS.decay, momentum=FLAGS.momentum)
    train_op = optimizer.apply_gradients(
            zip(grads, tvars),
            global_step = tf.contrib.framework.get_or_create_global_step())
    merged_summary = tf.summary.merge_all()

    train_and_val(train_op, loss_op, merged_summary, target_ph, gt_ph,
            file_names_placeholder, enqueue_op, q_close_op)


def main(_):
    """
    1. create graph
    2. train and eval
    """
    """get the inputs"""
    file_names_placeholder, enqueue_op, q_close_op, batch_img =\
            read_imgs(FLAGS.batch_size*FLAGS.sequence_length)
    """import VGG"""
    vgg_graph_def = tf.GraphDef()
    with open(FLAGS.vgg_model_frozen, "rb") as f:
        vgg_graph_def.ParseFromString(f.read())
    """the features"""
    features = tf.import_graph_def(vgg_graph_def, input_map={'inputs':
        batch_img}, return_elements=[FLAGS.feature_layer])[0]
    features_dim = features.get_shape().as_list()
    print('features_dim', features_dim)
    num_features = features_dim[1]*features_dim[2]
    """compress input dimensions"""
    w = tf.get_variable('input_compressor_w',
            shape=(1,1,features_dim[-1],128), dtype=tf.float32,
            initializer=tf.contrib.layers.xavier_initializer())
    features = tf.nn.conv2d(features, w, strides=(1,1,1,1), padding="VALID",
            name="input_compressor")
    """the tracker"""
    initializer = tf.random_uniform_initializer(-FLAGS.init_scale,FLAGS.init_scale)
    tracker = NTMTracker(FLAGS.sequence_length, FLAGS.batch_size,
            controller_num_layers=FLAGS.num_layers,
            initializer=initializer)
    inputs = tf.reshape(features, shape=[FLAGS.batch_size,
        FLAGS.sequence_length, -1], name="reshaped_inputs")
    #print('reshaped inputs:', inputs.get_shape())
    target_ph = tf.placeholder(tf.float32,
            shape=[FLAGS.batch_size, num_features], name="target")
    gt_ph = tf.placeholder(tf.float32,
        shape=[FLAGS.batch_size, FLAGS.sequence_length, num_features], name="ground_truth")
    tf.summary.image("ground_truth", tf.reshape(gt_ph,
        [-1,features_dim[1],features_dim[2],1]),
        max_outputs=FLAGS.batch_size*FLAGS.sequence_length)
    outputs, output_logits, states, debugs = tracker(inputs, target_ph)
    tf.summary.image("outputs", tf.reshape(outputs,
        [-1,features_dim[1],features_dim[2],1]),
        max_outputs=FLAGS.batch_size*FLAGS.sequence_length)
    #print('output_logits shape:', output_logits.get_shape())
    #output_logits is in [batch, seq_length, output_dim]
    #reshape it to [batch*seq_length, output_dim]
    """loss"""
    loss_op = tf.reduce_sum(
            tf.nn.softmax_cross_entropy_with_logits(
                tf.reshape(output_logits, [-1, num_features]),
                tf.nn.softmax(tf.reshape(gt_ph, [-1, num_features]))
                )) / (FLAGS.sequence_length *
                        FLAGS.batch_size)
    tf.summary.scalar('loss', loss_op)
    tf.summary.tensor_summary('outputs_summary', outputs)
    tf.summary.tensor_summary('output_logits_summary', output_logits)
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
            file_names_placeholder, enqueue_op, q_close_op, [outputs,
                output_logits, states, debugs])

if __name__ == '__main__':
    if (FLAGS.test_read_imgs):
        test_read_imgs()
    elif (FLAGS.lstm_only):
        lstm_only()
    else:
        tf.app.run()
