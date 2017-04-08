from __future__ import absolute_import

import tensorflow as tf
import numpy as np
import pickle
from datetime import datetime
import os
import scipy.misc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


from vgg import vgg_16

from ntm_tracker_new import NTMTracker, PlainNTMTracker, LoopNTMTracker
from ntm_cell import NTMCell
from ops import batched_smooth_cosine_similarity
from sklearn.decomposition import PCA
from receptive_field_sizes import conv43Points

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
flags.DEFINE_string("log_dir", "./log", "The log dir")
flags.DEFINE_integer("sequence_length", 20, "The length of input sequences")
flags.DEFINE_integer("model_length", 20, "The length of total steps of the tracker. Determines the physical length of the architecture in the graph. Affects the depth of back propagation in time. Longer input will be truncated")
flags.DEFINE_integer("batch_size", 16, "size of batch")
flags.DEFINE_string("feature_layer", "vgg_16/pool5/MaxPool:0", "The layer of feature to be put into NTM as input")
#flags.DEFINE_string("feature_layer", "vgg_16/conv4/conv4_3/Relu:0", "The layer of feature to be put into NTM as input")
flags.DEFINE_integer("max_gradient_norm", 5, "for gradient clipping normalization")
flags.DEFINE_float("learning_rate", 1e-4, "learning rate")
flags.DEFINE_float("momentum", 0.9, "learning rate")
flags.DEFINE_float("decay", 0.95, "learning rate")
flags.DEFINE_integer("hidden_size", 100, "number of LSTM cells")
flags.DEFINE_integer("num_layers", 10, "number of LSTM cells")
flags.DEFINE_string("tag", "", "tag for the log record")
flags.DEFINE_string("ckpt_path", "", "path for the ckpt file to be restored")
flags.DEFINE_integer("log_interval", 10, "number of epochs before log")
flags.DEFINE_float("init_scale", 0.05, "initial range for weights")
flags.DEFINE_integer("read_head_size", 3, "number of read heads")
flags.DEFINE_integer("write_head_size", 3, "number of write heads")
flags.DEFINE_boolean("two_step", False, "present the input in a 2-step manner")
flags.DEFINE_boolean("sequential", False, "present the input in a sequential manner")
flags.DEFINE_boolean("sevenbyseven", False, "present the input in a sequential manner, and with gt sevenbyseven")
flags.DEFINE_boolean("eightbyeight", False, "present the input in a sequential manner, and with gt eightbyeight")
flags.DEFINE_boolean("write_first", False, "write before read")
flags.DEFINE_boolean("sanity_check", False, "check if dataset is correct")
flags.DEFINE_boolean("sanity_check_compressor", False, "check if compressor is correct")
flags.DEFINE_boolean("sanity_check_trained_compressor", False, "check if compressor is correct")
flags.DEFINE_boolean("sanity_check_pca", False, "check if compressor is correct")
flags.DEFINE_boolean("compressor", False, "whether to use compressor.  If false, ignores compress_dim")
flags.DEFINE_boolean("copy_paste", False, "perform copy_paste task to check if the ntm is correct")
flags.DEFINE_integer("compress_dim", 128, "the output dimension of channels after input compression")
flags.DEFINE_float("bbox_crop_ratio", 5/float(7), "The indended width of bbox relative to the crop to be generated")
flags.DEFINE_integer("mem_size", 128, "size of mem")
flags.DEFINE_integer("mem_dim", 20, "dim of mem")
flags.DEFINE_boolean("test_input", False, "test the new get_input function")
flags.DEFINE_boolean("find_validation_batch", False, "find the actual validation batch by simulating shuffling")
flags.DEFINE_integer("gt_width", 7, "width of ground truth. a value of 7 means a 7x7 ground truth")
flags.DEFINE_integer("gt_depth", 8, "number of bytes used for each pixel")
flags.DEFINE_string("sequences_dir", "", "dir to look for sequences")
flags.DEFINE_integer("validation_interval", 100, "number of steps before validation")
flags.DEFINE_integer("validation_batch", 1, "validate only this number of batches")
flags.DEFINE_integer("min_skip_len", 1, "minimal number of frames to skip")
flags.DEFINE_integer("max_skip_len", 5, "maximal number of frames to skip")

FLAGS = flags.FLAGS

random.seed(42)

real_log_dir = os.path.abspath(os.path.join(FLAGS.log_dir,
    str(datetime.now())+FLAGS.tag))
print('real log dir: {}'.format(real_log_dir))

VGG_MEAN = tf.constant([123.68, 116.78, 103.94], dtype=tf.float32,
        shape=[1,1,3], name="VGG_MEAN")

def save_imgs(imgs, filename, savedir=real_log_dir):
    """
    Args:
        imgs: a list of ndarrays, each has shape [batch, length, w, h, c]
    """
    batch, length, _, _, _ = imgs[0].shape
    #print([x.shape for x in imgs])
    rows = len(imgs)*batch
    columns = length
    fig, axs = plt.subplots(rows, columns, figsize=(columns, rows), dpi=160)
    for batch_idx in xrange(batch):
        for set_idx, img in enumerate(imgs):
            for length_idx in xrange(length):
                ax = axs[batch_idx*len(imgs)+set_idx, length_idx]
                ax.imshow(np.squeeze(img[batch_idx, length_idx,:,:,:]))
                ax.axis('off')
    fig.savefig(os.path.join(savedir, filename+'.png'),
            bbox_inches='tight', pad_inches=0)
    plt.close(fig)

def get_valid_sequences(sequences_dir=FLAGS.sequences_dir, min_length=FLAGS.sequence_length, min_skip_len=FLAGS.min_skip_len, max_skip_len=FLAGS.max_skip_len):
    """dirs of sequences"""
    sequences = [os.path.join(sequences_dir,x) for x in
            sorted(os.listdir(sequences_dir))]
    result = []
    train = []
    val = []
    for seqdir in sequences:
        """the statistics files"""
        files = sorted([x[:-4] for x in os.listdir(seqdir) if
            x.endswith('.txt')])
        """
        Only retain the files that are long enough.
        For long sequences, dilate the steps taken to enrich the input
        """
        actual_max_skip_len = min(len(files) / min_length, max_skip_len)
        for skip in xrange(min_skip_len, actual_max_skip_len+1):
            sliced = files[::skip][:min_length]
            result.append((seqdir, sliced))
            if 'train' in seqdir:
                train.append((seqdir, sliced))
            elif 'val' in seqdir:
                val.append((seqdir, sliced))
            else:
                raise Exception('expect either train or val in sequence name')
    return result, train, val

def create_vgg(inputs, feature_layer):
    net, end_points = vgg_16(inputs)
    print(end_points.keys())
    return end_points[feature_layer]

def default_get_batch(index, batch_size, seq_length, seqs):
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

def sevenbyseven_get_batch(index, batch_size, seqs):
    """
    get a batch of frame names and their ground truths

    seqs: the sequence and their sub file names
    seq_length: the length of subsequence to take
    batch_size: the number of sequences to push into the batch
    """
    seq_batch = seqs[index:index+batch_size]
    index+=batch_size
    frame_names = []
    for seq, frames in seq_batch:
        frame_names += [os.path.join(seq, x) for x in frames]
    return frame_names, index

def get_input(batch_size):
    """
    read three things
    1. original image
    2. cropbox
    3. ground truth
    """
    """
    this is the filename without suffix
    """
    filename_nosuffix_ph = tf.placeholder(shape=(batch_size), dtype=tf.string)
    bin_suffix = tf.fill([batch_size], '.bin')
    txt_suffix = tf.fill([batch_size], '.txt')
    bins = tf.string_join([filename_nosuffix_ph, bin_suffix])
    txts = tf.string_join([filename_nosuffix_ph, txt_suffix])
    """process the txts"""
    txt_rdr = tf.TextLineReader()
    txt_q = tf.FIFOQueue(batch_size, tf.string)
    txt_enq_op = txt_q.enqueue_many(txts)
    txt_cls_op = txt_q.close()
    key, value = txt_rdr.read(txt_q)
    record_defaults = [[.0],[.0],[.0],[.0],[.0],[.0],[.0],[.0],['']]
    y1,x1,y2,x2,_,_,_,_,img_filename = tf.decode_csv(value, record_defaults)
    cropbox = tf.stack([y1,x1,y2,x2])
    cropboxes, img_filenames = tf.train.batch([cropbox, img_filename],
            batch_size = batch_size, num_threads=1)
    """process the imgs"""
    img_q = tf.FIFOQueue(batch_size, tf.string)
    img_enq_op = img_q.enqueue_many(img_filenames)
    img_cls_op = img_q.close()
    wf_rdr = tf.WholeFileReader()
    key, value = wf_rdr.read(img_q)
    my_img = tf.image.decode_jpeg(value)
    my_img = tf.reshape(tf.image.resize_images(my_img, [720, 1280]), (720, 1280, 3))
    my_img = my_img - VGG_MEAN
    batch_img = tf.train.batch([my_img],
            batch_size = batch_size,
            num_threads=1)
    batch_img = tf.image.crop_and_resize(batch_img, cropboxes,
            tf.range(batch_size), [224, 224])
    with tf.control_dependencies([txt_enq_op, img_enq_op]):
        batch_img = tf.identity(batch_img)
    """process the ground truths"""
    bin_q = tf.FIFOQueue(batch_size, tf.string)
    bin_enq_op = bin_q.enqueue_many(bins)
    bin_cls_op = bin_q.close()
    record_bytes = FLAGS.gt_width*FLAGS.gt_width*FLAGS.gt_depth
    fl_rdr = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    key, value = fl_rdr.read(bin_q)
    gt = tf.reshape(tf.cast(tf.decode_raw(value, tf.float64), tf.float32),
            [FLAGS.gt_width, FLAGS.gt_width])
    batch_gt = tf.train.batch([gt],
            batch_size = batch_size,
            num_threads = 1)
    with tf.control_dependencies([bin_enq_op]):
        batch_gt = tf.identity(batch_gt)
    close_qs_op = tf.group(txt_cls_op, img_cls_op, bin_cls_op)

    return filename_nosuffix_ph, batch_img, batch_gt, close_qs_op


def test_get_input(batch_size=1):
    filename_ph, batch_img, batch_gt, close_op = get_input(batch_size)
    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        imgs, gts = sess.run([batch_img, batch_gt], feed_dict={
            filename_ph:
            ['/home/jowos/data/ILSVRC2015/cropped/a/ILSVRC2015_train_00000000_0/000000']*batch_size
            })
        for idx in xrange(batch_size):
            scipy.misc.imsave('test_get_input_{}.png'.format(idx), imgs[idx])
            print(gts[idx])
        sess.run(close_op)
        coord.request_stop()
        coord.join(threads)


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
    return enqueue_placeholder, enqueue_op, queue_close_op, batch_img

def read_imgs_withbbox(batch_size):
    pass

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
        saver = tf.train.Saver(max_to_keep=1000)
        saver.restore(sess, "./vgg_16.ckpt")
        output = sess.run(features)
        print(output.shape)

        sess.run(queue_close_op) #close the queue
        coord.request_stop()
        coord.join(threads)

def train_and_val_sevenbyseven(#ops
        train_op, loss_op, q_close_op,
        #input placeholders
        file_names_placeholder, val_loss_ph,
        #terminal tensors,
        #summaries
        train_merged_summary,
        val_merged_summary,
        val_loss_summary,
        #tensors to plot to files
        saves,
        #global step variable
        global_step,
        get_batch):
    #check_op = tf.add_check_numerics_ops()
    with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
        print('session started')
        saver = tf.train.Saver(max_to_keep=1000)
        writer = tf.summary.FileWriter(real_log_dir, sess.graph)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        # initialize variables
        sess.run(tf.global_variables_initializer())
        if FLAGS.ckpt_path:
            saver.restore(sess, FLAGS.ckpt_path)
        #else:
        #    sess.run(tf.global_variables_initializer())
        print("start to run the training.")
        """
        1. get the statistics
        2. get the images
        3. extract the features
        4. train the network
        """
        #TODO: replace this
        print("getting valid sequences...")
        _, train_seqs, val_seqs = get_valid_sequences()
        print('{} sequences after length filtering'.format(
            len(train_seqs)+len(val_seqs)))
        #shuffle the order
        #divide train/test batches
        num_train = len(train_seqs)/FLAGS.batch_size*FLAGS.batch_size
        num_val = len(val_seqs)/FLAGS.batch_size*FLAGS.batch_size
        train_seqs = train_seqs[:num_train]
        val_seqs = val_seqs[:num_val]
        print('{} train seqs, {} val seqs'.format(
            len(train_seqs), len(val_seqs)))
        step = 0 #this is not global step, and is only relevant to logging
        num_epochs = FLAGS.num_epochs
        for epoch in xrange(num_epochs):
            print("training epoch {}".format(epoch))
            random.shuffle(train_seqs)
            print("shuffled training seqs")
            #train
            index = 0 #index used by get_batch
            while index < len(train_seqs):
                """validate first"""
                if step % FLAGS.validation_interval == 0:
                    print("{} validating...".format(step))
                    random.shuffle(val_seqs)
                    val_index = 0
                    accumu_loss = .0
                    count = 0
                    while val_index < len(val_seqs):
                        #get a batch
                        frame_names, index = get_batch(val_index, FLAGS.batch_size, val_seqs)
                        feed_dict = {file_names_placeholder:
                                    frame_names}
                        loss, summary, real_saves, gstep = sess.run(
                                [loss_op, val_merged_summary, saves,
                                    global_step],
                                feed_dict=feed_dict
                                )
                        accumu_loss += loss
                        writer.add_summary(summary, step)
                        save_imgs(real_saves,
                                'step_{}_validation_{}'.format(gstep,
                                    int(count)), real_log_dir)
                        count += 1
                        if count >= FLAGS.validation_batch:
                            break
                    accumu_loss /= float(count)
                    summary = sess.run(val_loss_summary,
                            feed_dict={val_loss_ph: accumu_loss})
                    writer.add_summary(summary, step)
                    print("{}: validation loss {}".format(step, accumu_loss))
                    save_path = saver.save(sess, os.path.join(real_log_dir,
                    "model.ckpt"), global_step=global_step)
                    print("model saved to {}".format(save_path))
                    with open("save_path.txt", "w") as f:
                        f.write(save_path)
                # this batch
                frame_names, index = get_batch(index,
                        FLAGS.batch_size, train_seqs)
                feed_dict = {file_names_placeholder:
                            frame_names}
                #now run the model
                """
                run: compute the output and train the model
                """
                loss, summary, _, real_saves, gstep = sess.run(
                        [loss_op, train_merged_summary, train_op, saves,
                            global_step],
                        feed_dict=feed_dict
                        )
                save_imgs(real_saves, 'step_{}_train'.format(gstep), real_log_dir)
                writer.add_summary(summary, step)
                if step % FLAGS.log_interval == 0:
                    print("{} training loss: {}".format(step, loss))
                """run a validation after certain number of steps"""
                step += 1
        print("{} validating...".format(step))
        random.shuffle(val_seqs)
        val_index = 0
        accumu_loss = .0
        count = 0
        while val_index < len(val_seqs):
            #get a batch
            frame_names, index = get_batch(val_index, FLAGS.batch_size, val_seqs)
            feed_dict = {file_names_placeholder:
                        frame_names}
            loss, summary, real_saves, gstep = sess.run(
                    [loss_op, val_merged_summary, saves,
                        global_step],
                    feed_dict=feed_dict
                    )
            accumu_loss += loss
            writer.add_summary(summary, step)
            save_imgs(real_saves,
                    'step_{}_validation_{}'.format(gstep,
                        int(count)), real_log_dir)
            count += 1
            if count >= FLAGS.validation_batch:
                break
        accumu_loss /= float(count)
        summary = sess.run(val_loss_summary,
                feed_dict={val_loss_ph: accumu_loss})
        writer.add_summary(summary, step)
        print("{}: validation loss {}".format(step, accumu_loss))
        save_path = saver.save(sess, os.path.join(real_log_dir,
        "model.ckpt"), global_step=global_step)
        print("model saved to {}".format(save_path))
        with open("save_path.txt", "w") as f:
            f.write(save_path)

        sess.run(q_close_op) #close the queue
        coord.request_stop()
        coord.join(threads)

def train_and_val_sequential(
        #ops
        train_op, loss_op, enqueue_op, q_close_op,
        #input placeholders
        file_names_placeholder, target_ph, gt_ph,
        train_merged_summary,
        val_merged_summary,
        global_step,
        other_ops=[],
        get_batch=default_get_batch):
    #check_op = tf.add_check_numerics_ops()
    with tf.Session() as sess:
        print('session started')
        saver = tf.train.Saver()
        writer = tf.summary.FileWriter(real_log_dir, sess.graph)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        # initialize variables
        if FLAGS.ckpt_path:
            saver.restore(sess, FLAGS.ckpt_path)
        else:
            sess.run(tf.global_variables_initializer())
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
            index = 0 #index used by get_batch
            while index < len(train_seqs):
                # this batch
                frame_names, real_gts, index = get_batch(index,
                        FLAGS.batch_size, FLAGS.sequence_length, train_seqs)
                feed_dict = {file_names_placeholder:
                            frame_names}
                #print(feed_dict)
                sess.run(enqueue_op, feed_dict=feed_dict)
                #now run the model
                """
                run: compute the output and train the model
                """
                feed_dict = {
                        target_ph: real_gts[:,0,:],
                        gt_ph: real_gts
                        }
                loss, summary, _ = sess.run(
                        [loss_op, train_merged_summary, train_op],
                        feed_dict=feed_dict
                        )
                writer.add_summary(summary, step)
                if step % FLAGS.log_interval == 0:
                    print("{} training loss: {}".format(step, loss))
                step += 1

        index = 0
        step = 0
        while index < len(test_seqs):
            #get a batch
            frame_names, real_gts, index = get_batch(index, FLAGS.batch_size,
                    FLAGS.sequence_length, test_seqs)
            feed_dict = {file_names_placeholder:
                        frame_names}
            sess.run(enqueue_op, feed_dict=feed_dict)
            feed_dict = {
                    target_ph: real_gts[:,0,:],
                    gt_ph: real_gts
                    }
            loss, summary = sess.run(
                    [loss_op, val_merged_summary],
                    feed_dict=feed_dict
                    )
            writer.add_summary(summary, step)
            print("{}: validation loss {}".format(step, loss))
            step += 1

        save_path = saver.save(sess, os.path.join(real_log_dir,
            "model.ckpt"), global_step=global_step)
        print("model saved to {}".format(save_path))
        with open("save_path.txt", "w") as f:
            f.write(save_path)

        sess.run(q_close_op) #close the queue
        coord.request_stop()
        coord.join(threads)

def train_and_val(train_op, loss, merged, target, gt,
        file_names_placeholder, enqueue_op, q_close_op, other_ops=[],
        get_batch=default_get_batch):
    #check_op = tf.add_check_numerics_ops()
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
            shape=(1,1,features_dim[-1],FLAGS.compress_dim), dtype=tf.float32,
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
        max_outputs=FLAGS.batch_size*FLAGS.sequence_length*(2 if
            FLAGS.two_step else 1))
    """actually build the lstm"""
    print("building lstm")
    outputs = []
    state = initial_state
    with tf.variable_scope("lstm-tracker"):
        for time_step in range(FLAGS.sequence_length):
            if time_step > 0:
                tf.get_variable_scope().reuse_variables()
                cell_output, state = cell(
                    tf.concat([inputs[:, time_step, :], dummy_target],1), state)
            else:
                cell_output, state = cell(
                    tf.concat([inputs[:, time_step, :], target_ph],1), state)
            outputs.append(cell_output)
    output = tf.reshape(tf.concat(outputs, 1), [-1, FLAGS.hidden_size])
    """compress the output to our desired dimensions"""
    softmax_w = tf.get_variable(
        "softmax_w", [FLAGS.hidden_size, num_features], dtype=tf.float32)
    softmax_b = tf.get_variable("softmax_b", [num_features], dtype=tf.float32)
    output_logits = tf.matmul(output, softmax_w) + softmax_b
    tf.summary.image("outputs", tf.reshape(tf.nn.softmax(output_logits),
        [-1,features_dim[1],features_dim[2],1]),
        max_outputs=FLAGS.batch_size*FLAGS.sequence_length*(2 if
            FLAGS.two_step else 1))
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

def ntm():
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
            shape=(1,1,features_dim[-1],FLAGS.compress_dim), dtype=tf.float32,
            initializer=tf.contrib.layers.xavier_initializer())
    features = tf.nn.conv2d(features, w, strides=(1,1,1,1), padding="VALID",
            name="input_compressor")
    """the tracker"""
    initializer = tf.random_uniform_initializer(-FLAGS.init_scale,FLAGS.init_scale)
    tracker = NTMTracker(FLAGS.sequence_length, FLAGS.batch_size,
            num_features, controller_num_layers=FLAGS.num_layers,
            initializer=initializer, read_head_size=FLAGS.read_head_size,
            write_head_size=FLAGS.write_head_size, two_step=FLAGS.two_step,
            write_first=FLAGS.write_first,
            controller_hidden_size=FLAGS.hidden_size
            )
    inputs = tf.reshape(features, shape=[FLAGS.batch_size,
        FLAGS.sequence_length, -1], name="reshaped_inputs")
    #print('reshaped inputs:', inputs.get_shape())
    target_ph = tf.placeholder(tf.float32,
            shape=[FLAGS.batch_size, num_features], name="target")
    """
    ground truth
    """
    gt_ph = tf.placeholder(tf.float32,
        shape=[FLAGS.batch_size, FLAGS.sequence_length, num_features], name="ground_truth")
    tf.summary.image("ground_truth", tf.reshape(gt_ph,
        [-1,features_dim[1],features_dim[2],1]),
        max_outputs=FLAGS.batch_size*FLAGS.sequence_length)
    """
    build the tracker
    """
    outputs, output_logits, states, debugs = tracker(inputs, target_ph)
    tf.summary.image("outputs", tf.reshape(tf.sigmoid(output_logits),
        [-1,features_dim[1],features_dim[2],1]),
        max_outputs=(FLAGS.batch_size*FLAGS.sequence_length if
            not FLAGS.two_step else FLAGS.batch_size*(2*FLAGS.sequence_length-1)))
    #print('output_logits shape:', output_logits.get_shape())
    #output_logits is in [batch, seq_length, output_dim]
    #reshape it to [batch*seq_length, output_dim]
    """loss"""
    loss_op = tf.reduce_sum(
        tf.nn.softmax_cross_entropy_with_logits(
            logits=tf.reshape(output_logits, [-1, num_features]),
            labels=tf.nn.softmax(tf.reshape(gt_ph, [-1, num_features]))
            )) / (FLAGS.sequence_length *
                    FLAGS.batch_size * (2 if FLAGS.two_step else 1))
    tf.summary.scalar('loss', loss_op)
    tf.summary.tensor_summary('outputs_summary', outputs)
    tf.summary.tensor_summary('output_logits_summary', output_logits)
    """training op"""
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(loss_op, tvars),
            FLAGS.max_gradient_norm)
    optimizer = tf.train.RMSPropOptimizer(FLAGS.learning_rate,
            decay=FLAGS.decay, momentum=FLAGS.momentum)
    train_op = optimizer.apply_gradients(
            zip(grads, tvars),
            global_step = tf.contrib.framework.get_or_create_global_step())
    merged_summary = tf.summary.merge_all()

    return (train_op, loss_op, merged_summary, target_ph, gt_ph,
            file_names_placeholder, enqueue_op, q_close_op, [outputs,
                output_logits, states, debugs], default_get_batch)

def ntm_two_step():
    """
    1. create graph
    so we want to increase the "target indicator" dimension and "label"
    dimension by one so that [000000...001] can be used to signify
    background
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
            shape=(1,1,features_dim[-1],FLAGS.compress_dim), dtype=tf.float32,
            initializer=tf.contrib.layers.xavier_initializer())
    features = tf.nn.conv2d(features, w, strides=(1,1,1,1), padding="VALID",
            name="input_compressor")
    """the tracker"""
    initializer = tf.random_uniform_initializer(-FLAGS.init_scale,FLAGS.init_scale)
    tracker = NTMTracker(FLAGS.sequence_length, FLAGS.batch_size,
            num_features+1, controller_num_layers=FLAGS.num_layers,
            initializer=initializer, read_head_size=FLAGS.read_head_size,
            write_head_size=FLAGS.write_head_size, two_step=FLAGS.two_step,
            write_first=FLAGS.write_first,
            controller_hidden_size=FLAGS.hidden_size
            )
    inputs = tf.reshape(features, shape=[FLAGS.batch_size,
        FLAGS.sequence_length, -1], name="reshaped_inputs")
    #print('reshaped inputs:', inputs.get_shape())
    target_ph = tf.placeholder(tf.float32,
            shape=[FLAGS.batch_size, num_features], name="target")
    """
    ground truth
    +1 for the "background"
    """
    gt_ph = tf.placeholder(tf.float32,
        shape=[FLAGS.batch_size, FLAGS.sequence_length, num_features], name="ground_truth")
    tf.summary.image("ground_truth", tf.reshape(gt_ph,
        [-1,features_dim[1],features_dim[2],1]),
        max_outputs=FLAGS.batch_size*FLAGS.sequence_length)

    """
    remove the first frame ground truth
    and pad the ground truth to twice the sequence length - 2
    """
    assert(FLAGS.sequence_length >= 2, "two_step must be used with sequence at least length 2")
    gt_pad = tf.zeros_like(gt_ph[:,1:,:], dtype=tf.float32, name="gt_pad")
    gt_pad_bg_bit = tf.expand_dims(tf.ones_like(gt_pad[:,:,1], dtype=tf.float32,
            name="gt_pad_bg_bit"), -1)
    gt_bg_bit = tf.expand_dims(tf.zeros_like(gt_pad[:,:,1], dtype=tf.float32,
            name="gt_bg_bit"), -1)
    gt_pad_augmented = tf.concat([gt_pad, gt_pad_bg_bit], axis=2,
            name="gt_pad_augmented")
    gt_ph_augmented = tf.concat([gt_ph[:,1:,:], gt_bg_bit], axis=2,
            name="gt_augmented")
    gt_stacked = tf.stack((gt_pad_augmented, gt_ph_augmented), axis=2)
    labels = tf.reshape(gt_stacked, [FLAGS.batch_size,
        FLAGS.sequence_length*2-2, num_features+1])
    """
    now prepend the ground truth for the zeroth frame
    """
    first_frame_gt = tf.concat([
        tf.zeros([FLAGS.batch_size, 1, num_features]),
        tf.ones([FLAGS.batch_size, 1, 1])], axis=2,
        name="gt_first_frame")
    labels=tf.concat([first_frame_gt, labels], axis=1, name="labels")
    tf.summary.image("labels", tf.reshape(labels,
        [-1,2*FLAGS.sequence_length-1,num_features+1,1]),
        max_outputs=FLAGS.batch_size)
    """
    build the tracker
    """
    outputs, output_logits, states, debugs = tracker(inputs, target_ph)
    tf.summary.image("outputs", tf.reshape(outputs,
        [-1,2*FLAGS.sequence_length-1,num_features+1,1]),
        max_outputs=FLAGS.batch_size)
    #print('output_logits shape:', output_logits.get_shape())
    #output_logits is in [batch, seq_length, output_dim]
    #reshape it to [batch*seq_length, output_dim]
    """loss"""
    loss_op = tf.reduce_sum(
        tf.nn.softmax_cross_entropy_with_logits(
            logits=tf.reshape(output_logits, [-1, num_features+1]),
            labels=tf.nn.softmax(tf.reshape(labels, [-1, num_features+1]))
            )) / ((2*FLAGS.sequence_length-1) * FLAGS.batch_size)
    #"""l2 loss"""
    #labels=tf.reshape(labels, [-1, num_features])
    #logits=tf.reshape(tf.sigmoid(output_logits), [-1, num_features]),
    #loss_op = tf.nn.l2_loss(logits-labels) /\
    #        FLAGS.batch_size*(2*FLAGS.sequence_length-1)
    tf.summary.scalar('loss', loss_op)
    tf.summary.tensor_summary('outputs_summary', outputs)
    tf.summary.tensor_summary('output_logits_summary', output_logits)
    """training op"""
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(loss_op, tvars),
            FLAGS.max_gradient_norm)
    optimizer = tf.train.RMSPropOptimizer(FLAGS.learning_rate,
            decay=FLAGS.decay, momentum=FLAGS.momentum)
    train_op = optimizer.apply_gradients(
            zip(grads, tvars),
            global_step = tf.contrib.framework.get_or_create_global_step())
    merged_summary = tf.summary.merge_all()

    return (train_op, loss_op, merged_summary, target_ph, gt_ph,
            file_names_placeholder, enqueue_op, q_close_op, [outputs,
                output_logits, states, debugs], default_get_batch)

def ntm_sequential():
    """
    sequential means instead of presenting the whole feature map at once, I
    present each feature one by one
    1. create graph
    """
    """get the inputs"""
    train_summaries = []
    val_summaries = []
    file_names_placeholder, enqueue_op, q_close_op, batch_img =\
            read_imgs(FLAGS.batch_size*FLAGS.sequence_length)
    train_summaries.append(tf.summary.image('train_batch_img', batch_img,
        max_outputs=FLAGS.batch_size))
    val_summaries.append(tf.summary.image('val_batch_img', batch_img,
        max_outputs=FLAGS.batch_size))
    """import VGG"""
    vgg_graph_def = tf.GraphDef()
    with open(FLAGS.vgg_model_frozen, "rb") as f:
        vgg_graph_def.ParseFromString(f.read())
    """the features"""
    features = tf.import_graph_def(vgg_graph_def, input_map={'inputs':
        batch_img}, return_elements=[FLAGS.feature_layer])[0]
    features_dim = features.get_shape().as_list()
    num_channels = features_dim[-1]
    print('features_dim', features_dim)
    num_features = features_dim[1]*features_dim[2]
    if FLAGS.compressor:
        """compress input dimensions"""
        w = tf.get_variable('input_compressor_w',
                shape=(1,1,features_dim[-1],FLAGS.compress_dim), dtype=tf.float32,
                initializer=tf.contrib.layers.xavier_initializer())
        features = tf.nn.conv2d(features, w, strides=(1,1,1,1), padding="VALID",
                name="input_compressor")
        num_channels = FLAGS.compress_dim
    print("num_channels:", num_channels)
    """
    the inputs;
    features is of shape [batch * seq_length, 28, 28, 128]
    originally it's reshaped to [batch, seq_len, num_features*num_channels]
    now we want it to be [batch, seq_len*num_features, 128]
    """

    inputs = tf.reshape(features, shape=[FLAGS.batch_size,
        FLAGS.sequence_length, num_features, num_channels], name="reshaped_inputs")
    #print('reshaped inputs:', inputs.get_shape())
    """
    placeholder to accept target indicator input
    because it's only for the 0th frame, so it's 2d
    """
    target_ph = tf.placeholder(tf.float32,
            shape=[FLAGS.batch_size, num_features], name="target")
    """
    build the tracker inputs
    the inputs should be a matrix of [batch_size, xxx, 128+1+1]
    xxx:
        [0:num_features]: first frame, all features
        [num_features:num_features+1]: frame delimiter, [129]=1
        [num_features+1:num_features+2]: second frame, first feature
        [num_features+2:num_features+3]: second frame, feature delimiter, [128]=1
        [num_features+3:num_features+4]: second frame, second feature
        [num_features+4:num_features+5]: second frame, feature delimiter, [128]=1
        ...
    there will ultimately be
    num_features + (sequence_length - 1) * (1 + 2 * num_features) steps
    """
    total_steps = num_features + (FLAGS.sequence_length - 1) * (2 * num_features + 1)
    print("constructing inputs...")
    #shape [batch, seq_len, num_features, 130]
    inputs_padded = tf.concat([inputs, tf.zeros([FLAGS.batch_size,
        FLAGS.sequence_length, num_features, 2])], 3)
    #shape [batch, sequence_length-1, num_features, 130]
    inputs_no_zeroth = inputs_padded[:, 1:, :, :]
    #shape [batch, 1, 1, 128]
    dummy_feature = tf.zeros([FLAGS.batch_size, 1, 1, num_channels])
    #shape [batch, 1, 1, 130]
    frame_delimiter = tf.concat([
            dummy_feature,
            tf.zeros([FLAGS.batch_size, 1, 1, 1], dtype=tf.float32),
            tf.ones([FLAGS.batch_size, 1, 1, 1], dtype=tf.float32),
            ], 3)
    #frame delimiters, number: sequence_length - 1
    #shape [batch, sequence_length - 1, 1, 130]
    frame_delimiters = tf.tile(frame_delimiter,
            [1, FLAGS.sequence_length-1, 1, 1],
            name="frame_delimiters")
    feature_delimiter = tf.concat([
            dummy_feature,
            tf.ones([FLAGS.batch_size, 1, 1, 1], dtype=tf.float32),
            tf.zeros([FLAGS.batch_size, 1, 1, 1], dtype=tf.float32),
            ], 3)
    #feature delimiters, number: 1 per feature
    #shape [batch, sequence_length-1, num_features, 130]
    feature_delimiters = tf.tile(feature_delimiter,
            [1, FLAGS.sequence_length-1, num_features, 1],
            name="feature_delimiters")
    #now insert the feature delimiters
    inputs_no_zeroth = tf.reshape(tf.concat(
            [inputs_no_zeroth, feature_delimiters], 3),
            [FLAGS.batch_size, FLAGS.sequence_length-1, num_features*2,
                num_channels+2])
    #now insert the frame delimiters
    inputs_no_zeroth = tf.concat(
            [frame_delimiters, inputs_no_zeroth], 2)
    #now add back the zeroth frame
    inputs_no_zeroth = tf.reshape(inputs_no_zeroth,
            [
                FLAGS.batch_size,
                (FLAGS.sequence_length-1)*(2*num_features+1),
                num_channels+2])
    """
    num_features + (sequence_length - 1) * (1 + 2 * num_features) steps
    """
    inputs = tf.concat([
            inputs_padded[:,0,:,:],
            inputs_no_zeroth,
            ], 1, name="serial_inputs")
    target = tf.concat([
            target_ph,
            tf.zeros([FLAGS.batch_size,
                (FLAGS.sequence_length - 1) * (2 * num_features + 1),
                ], dtype=tf.float32)], 1)
    #dims: [batch_size, total_steps, 131]
    inputs = tf.concat([
        inputs,
        tf.expand_dims(target, -1)], -1)
    print("constructing ground truths...")
    """
    ground truth
    gt_ph is supposed to be fed with ground truths directly extracted from
    input batcher
    """
    gt_ph = tf.placeholder(tf.float32,
        shape=[FLAGS.batch_size, FLAGS.sequence_length, num_features], name="ground_truth")
    #tf.summary.image("ground_truth", tf.reshape(gt_ph,
    #    [-1,features_dim[1],features_dim[2],1]),
    #    max_outputs=FLAGS.batch_size*FLAGS.sequence_length)
    """
    there will be
    num_features + (sequence_length - 1) * (1 + 2 * num_features) steps

    how to produce?
    1. remove the first frame
    2. pad the features with num_features zeros
    3. pad the features with 1 zero at beginning
    4. pad at the beginning num_features zeros
    """

    """
    remove the first frame ground truth and create pad
    the dimension for gt_ph [batch_size, seq_length, num_features]
    """
    assert(FLAGS.sequence_length >= 2, "two_step must be used with sequence at least length 2")
    gt_pad = tf.zeros_like(gt_ph[:,1:,:], dtype=tf.float32, name="gt_pad")
    """
    stack at last axis, so that every feature scalar is prepended by a zero
    scalar
    """
    gt = gt_ph[:,1:,:]
    reshape_gt_ph = tf.reshape(gt_ph[:,1:,:],
        [FLAGS.batch_size*(FLAGS.sequence_length-1),features_dim[1],features_dim[2],1])
    train_summaries.append(tf.summary.image("train_ground_truth",
        reshape_gt_ph,
        max_outputs=FLAGS.batch_size*(FLAGS.sequence_length-1)))
    val_summaries.append(tf.summary.image("val_ground_truth",
        reshape_gt_ph,
        max_outputs=FLAGS.batch_size*(FLAGS.sequence_length-1)))
    gt_stacked = tf.stack((gt_pad, gt_ph[:,1:,:]), axis=3)
    labels = tf.reshape(gt_stacked, [FLAGS.batch_size,
        FLAGS.sequence_length-1, 2*num_features])
    """
    prepend each sequence with 1 zero, for the sequence delimiter
    """
    labels = tf.concat([
        tf.zeros([FLAGS.batch_size, FLAGS.sequence_length-1, 1]),
        labels], 2)
    labels = tf.reshape(labels,
            [FLAGS.batch_size, (FLAGS.sequence_length-1)*(2*num_features+1)])
    """
    now prepend the ground truth for the zeroth frame
    """
    first_frame_gt = tf.zeros([FLAGS.batch_size, num_features],
            name="gt_first_frame")
    labels=tf.concat([first_frame_gt, labels], axis=1, name="labels")
    #tf.summary.image("labels", tf.reshape(labels,
    #    [1,FLAGS.batch_size,num_features+(FLAGS.sequence_length-1)*(2*num_features+1),1]),
    #    max_outputs=1)
    labels = tf.expand_dims(labels, -1)

    print("constructing tracker...")
    """the tracker"""
    initializer = tf.random_uniform_initializer(-FLAGS.init_scale,FLAGS.init_scale)
    tracker = LoopNTMTracker(total_steps, 1,
            initializer,
            mem_size=FLAGS.mem_size, mem_dim=FLAGS.mem_dim,
            controller_num_layers=FLAGS.num_layers,
            controller_hidden_size=FLAGS.hidden_size,
            read_head_size=FLAGS.read_head_size,
            write_head_size=FLAGS.write_head_size,
            write_first=FLAGS.write_first,)
    """
    shape of outputs: [batch, model_length, 1]
    """
    print(inputs.get_shape().as_list())
    outputs, output_logits, Ms, ws, reads = tracker(inputs)
    print(output_logits.get_shape().as_list())
    """
    add summaries
    """
    print(Ms.get_shape().as_list())
    reshape_Ms = tf.reshape(Ms,
        [FLAGS.batch_size, FLAGS.mem_size, FLAGS.mem_dim*total_steps, 1])
    train_summaries.append(tf.summary.image('train_M', reshape_Ms,
        max_outputs=FLAGS.batch_size))
    val_summaries.append(tf.summary.image('val_M', reshape_Ms,
        max_outputs=FLAGS.batch_size))
    """w"""
    print(ws.get_shape().as_list())
    reshape_w_reads = tf.reshape(ws[:,:FLAGS.read_head_size,:,:],
            [FLAGS.batch_size, FLAGS.mem_size*FLAGS.read_head_size, total_steps, 1])
    reshape_w_writes = tf.reshape(ws[:,FLAGS.read_head_size:,:,:],
            [FLAGS.batch_size, FLAGS.mem_size*FLAGS.read_head_size, total_steps, 1])
    train_summaries.append(tf.summary.image('train_w_reads', reshape_w_reads,
            max_outputs=FLAGS.batch_size))
    train_summaries.append(tf.summary.image('train_w_writes', reshape_w_writes,
            max_outputs=FLAGS.batch_size))
    val_summaries.append(tf.summary.image('val_w_reads', reshape_w_reads,
            max_outputs=FLAGS.batch_size))
    val_summaries.append(tf.summary.image('val_w_writes', reshape_w_writes,
            max_outputs=FLAGS.batch_size))
    """reads"""
    reshape_reads = tf.reshape(reads, [FLAGS.batch_size*FLAGS.read_head_size,
                FLAGS.mem_dim, total_steps, 1])
    train_summaries.append(tf.summary.image('train_reads', reshape_reads,
            max_outputs=FLAGS.batch_size*FLAGS.read_head_size))
    val_summaries.append(tf.summary.image('val_reads', reshape_reads,
            max_outputs=FLAGS.batch_size*FLAGS.read_head_size))

    """
    now the subgraph to convert model output sequence to perceivable heatmaps
    """
    output_gather = tf.squeeze(output_logits, axis=2)
    """remove the output for first frame"""
    output_gather = output_gather[:,num_features:]
    output_first_frame = output_gather[:, :num_features]
    """remove the output for sequence delimiter"""
    output_gather = tf.reshape(output_gather, [FLAGS.batch_size,
        FLAGS.sequence_length-1, 2*num_features+1])
    output_sequence_delimiter = output_gather[:,:,:1]
    output_gather = output_gather[:,:,1:]
    """remove the output of first step in 2-step presentation"""
    output_gather = tf.reshape(output_gather, [FLAGS.batch_size,
        FLAGS.sequence_length-1, num_features, 2])
    output_first_step = output_gather[:,:,:,0]
    output_gather = output_gather[:,:,:,1]
    other_outputs = tf.concat([
        tf.reshape(output_first_frame, [-1]),
        tf.reshape(output_sequence_delimiter, [-1]),
        tf.reshape(output_first_step, [-1])])
    output_sigmoids = tf.sigmoid(output_gather)
    reshape_output_gather = tf.reshape(output_sigmoids,
                [FLAGS.batch_size*(FLAGS.sequence_length-1),
                    features_dim[1],features_dim[2],1])
    train_summaries.append(tf.summary.image("train_gathered_outputs",
            reshape_output_gather,
            max_outputs=FLAGS.batch_size*(FLAGS.sequence_length-1)))
    val_summaries.append(tf.summary.image("val_gathered_outputs",
            reshape_output_gather,
            max_outputs=FLAGS.batch_size*(FLAGS.sequence_length-1)))
    #print('output_logits shape:', output_logits.get_shape())
    #output_logits is in [batch, seq_length, output_dim]
    #reshape it to [batch*seq_length, output_dim]
    #"""loss"""
    loss_op = tf.reduce_sum(
        tf.nn.softmax_cross_entropy_with_logits(
            logits=tf.reshape(output_gather, [-1, num_features]),
            labels=tf.reshape(gt, [-1, num_features])
            )) / (FLAGS.sequence_length-1)\
        #+ tf.losses.log_loss(tf.zeros_like(other_outputs), tf.sigmoid(other_outputs))
    print("constructing loss...")
    """log loss"""
    #loss_op = tf.losses.log_loss(, tf.reshape(output_sigmoids, [
    #    FLAGS.batch_size*(FLAGS.sequence_length-1), num_features
    #    ]))
    train_summaries.append(tf.summary.scalar('train_loss', loss_op))
    val_summaries.append(tf.summary.scalar('val_loss', loss_op))
    """training op"""
    tvars = tf.trainable_variables()
    # the gradient tensors
    global_step = tf.Variable(0, name='global_step', trainable=False)
    grads, _ = tf.clip_by_global_norm(tf.gradients(loss_op, tvars),
            FLAGS.max_gradient_norm)
    optimizer = tf.train.RMSPropOptimizer(FLAGS.learning_rate,
            decay=FLAGS.decay, momentum=FLAGS.momentum)
    train_op = optimizer.apply_gradients(
            zip(grads, tvars),
            global_step = global_step)
    train_merged_summary = tf.summary.merge(train_summaries)
    val_merged_summary = tf.summary.merge(val_summaries)

    return (#ops
            train_op, loss_op, enqueue_op, q_close_op,
            #input placeholders
            file_names_placeholder, target_ph, gt_ph,
            #terminal tensors,
            #summaries
            train_merged_summary,
            val_merged_summary,
            #global step variable
            global_step,
            #other ops
            [outputs, output_logits],
            #get batch function
            default_get_batch)


def resize_imgs(batch_img, bboxes, bbox_grid, crop_grid):
    boxes = tf.stack(
            [calculate_crop_box(bbox, bbox_grid, crop_grid)
                for bbox in bboxes],
            axis=0)
    ind = tf.range(0, batch_img.get_shape().as_list()[0])
    crop_size = batch_img.get_shape().as_list()[1:3]
    return tf.image.crop_and_resize(batch_img, boxes, ind, crop_size)

def calculate_crop_box(bbox, bbox_grid, crop_grid):
    """
    Args:
        bbox: a 1D 4-tensor (xmin, xmax, ymin, ymax) of normalized coordinate
        bbox_grid: a 2-tensor (row_grid, column_grid)
        crop_grid: a 2-tensor (row_grid, column_grid)
    """
    width = (bbox[1] - bbox[0]) / bbox_grid[1] * crop_grid[1]
    height = (bbox[3] - bbox[2]) / bbox_grid[0] * crop_grid[0]

    xcenter = (bbox[0] + bbox[1]) / 2
    ycenter = (bbox[2] + bbox[3]) / 2

    xmin = xcenter - width / 2
    xmax = xcenter + width / 2
    ymin = ycenter - height / 2
    ymax = ycenter + height / 2

    return tf.constant([ymin, xmin, ymax, xmax])

def ntm_active_resize():
    """
    implement the module with active resizing
    """
    """
    get the inputs
    the first dimension of batch_img here is [batch_size*sequence_length]
    the same for bboxes
    """
    file_names_placeholder, enqueue_op, q_close_op, batch_img, bboxes =\
            read_imgs_withbbox(FLAGS.batch_size*FLAGS.sequence_length)
    batch_img = tf.reshape(batch_img, [FLAGS.batch_size, FLAGS.sequence_length,
        224, 224, 3])
    bboxes = tf.reshape(bboxes, [FLAGS.batch_size, FLAGS.sequence_length, 4])

    """import VGG"""
    vgg_graph_def = tf.GraphDef()
    with open(FLAGS.vgg_model_frozen, "rb") as f:
        vgg_graph_def.ParseFromString(f.read())
    """
    the features. the network is not actually used. only statistics of
    crop_grid and bbox_grid are needed
    """
    dummy_features = tf.import_graph_def(vgg_graph_def, return_elements=[FLAGS.feature_layer])[0]
    features_dim = dummy_features.get_shape().as_list()
    crop_grid = tf.constant([features_dim[1], features_dim[2]],
            dtype=tf.float32, name="crop_grid")
    bbox_grid = tf.constant([
        round(FLAGS.bbox_crop_ratio*features_dim[1]),
        round(FLAGS.bbox_crop_ratio*features_dim[2])],
            dtype=tf.float32, name="bbox_grid")

    """
    the ntm cell
    """
    initializer = tf.random_uniform_initializer(-FLAGS.init_scale,FLAGS.init_scale)
    cell = NTMCell(1, controller_num_layers=FLAGS.num_layers,
            controller_hidden_size=FLAGS.hidden_size,
            read_head_size=FLAGS.read_head_size,
            write_head_size=FLAGS.write_head_size,
            write_first=FLAGS.write_first,)

    """
    build the tracker:
        1. divide the input into separate time steps
        2. pre-precess the input of each time step
        3. feed the input, and get output
        4. set the bbox resize parameters of the next frame using the output
    """
    print("constructing tracker...")
    with tf.variable_scope('ntm-tracker', initializer=initializer):
        # set the initial states here
        outputs = []
        output_logits = []
        states = []
        debugs = []
        state = cell.zero_state(FLAGS.batch_size)
        states.append(state)
        this_batch_bboxes = bboxes[:,0,:]
        for idx in xrange(FLAGS.sequence_length):
            if idx > 0:
                tf.get_variable_scope().reuse_variables()
            #extract the input image batch of this time step
            this_batch_imgs = batch_img[:,idx,:,:,:]
            #preprocess the input
            resized_batch = resize_imgs(this_batch_imgs, this_batch_bboxes, bbox_grid,
                    crop_grid)
            #feed through VGG
            features = tf.import_graph_def(vgg_graph_def, input_map={'inputs':
                resized_batch}, return_elements=[FLAGS.feature_layer])[0]
            #compress the features
            w = tf.get_variable('input_compressor_w',
                    shape=(1,1,features_dim[-1],FLAGS.compress_dim), dtype=tf.float32,
                    initializer=tf.contrib.layers.xavier_initializer())
            features = tf.nn.conv2d(features, w, strides=(1,1,1,1), padding="VALID",
                    name="input_compressor")
            #build the cell


    """
    the resize module
    The bbox_ph will be replaced by actual decoded bbox outputs from the tracker
    """
    bbox_ph = tf.placeholder(tf.float32, shape=bboxes.get_shape())
    resized_imgs = resize_imgs(batch_img, bbox_ph)

    """import VGG"""
    vgg_graph_def = tf.GraphDef()
    with open(FLAGS.vgg_model_frozen, "rb") as f:
        vgg_graph_def.ParseFromString(f.read())
    """the features"""
    features = tf.import_graph_def(vgg_graph_def, input_map={'inputs':
        batch_img}, return_elements=[FLAGS.feature_layer])[0]
    features_dim = features.get_shape().as_list()
    crop_grid = tf.constant([features_dim[1], features_dim[2]],
            dtype=tf.float32, name="crop_grid")
    bbox_grid = tf.constant([
        round(FLAGS.bbox_crop_ratio*features_dim[1]),
        round(FLAGS.bbox_crop_ratio*features_dim[2])],
            dtype=tf.float32, name="bbox_grid")


def sanity_check_compressor(ckpt_path='/tmp/ntm-tracker/2017-02-18 11:28:18.000892batchsize16-seqlen2-numlayer10-hidden400-epoch100-lr1e-2-rw10-2step-write1st-augmentegt/model.ckpt',
        compressor=False, trained=False, pca=False):
    """
    instead of training a real NTM, try to make sure we can generate heat maps
    by dot producting the features
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
    num_features = features_dim[1]*features_dim[2]
    #num_channels = features_dim[3]
    """the compressor"""
    if pca:
        pca_features = tf.placeholder(tf.float32,
                shape=[FLAGS.batch_size, FLAGS.sequence_length,
                    num_features, FLAGS.compress_dim])
    if compressor:
        w = tf.get_variable('input_compressor_w',
                shape=(1,1,features_dim[-1],FLAGS.compress_dim), dtype=tf.float32,
                initializer=tf.contrib.layers.xavier_initializer())
        features = tf.nn.conv2d(features, w, strides=(1,1,1,1), padding="VALID",
                name="input_compressor")
        if trained:
            saver = tf.train.Saver({'input_compressor_w': w})
    #the features [batch*length, 28, 28, 512]
    features = tf.reshape(features,
            [FLAGS.batch_size, FLAGS.sequence_length, num_features,
                -1])
    gt_ph = tf.placeholder(tf.float32,
        shape=[FLAGS.batch_size, FLAGS.sequence_length, num_features], name="ground_truth")
    # [batch, 1, compressdim]
    if pca:
        first_frame_feature = tf.matmul(
                tf.expand_dims(gt_ph[:,0,:], -1),
                pca_features[:,0,:,:], transpose_a=True)
    # [batch, 1, seq_len*num_features]
        similarity = batched_smooth_cosine_similarity(
                tf.reshape(pca_features, [FLAGS.batch_size,
                    FLAGS.sequence_length*num_features, -1]),
                first_frame_feature)
    else:
        first_frame_feature = tf.matmul(
                tf.expand_dims(gt_ph[:,0,:], -1),
                features[:,0,:,:], transpose_a=True)
        similarity = batched_smooth_cosine_similarity(
                tf.reshape(features, [FLAGS.batch_size,
                    FLAGS.sequence_length*num_features, -1]),
                first_frame_feature)
    similarity = tf.reshape(similarity, [FLAGS.batch_size,
        FLAGS.sequence_length, num_features])
    similarity_summary = tf.summary.image("similarity", tf.reshape(similarity,
        [FLAGS.batch_size*FLAGS.sequence_length, features_dim[1],
            features_dim[2], 1]),
        max_outputs=FLAGS.batch_size*FLAGS.sequence_length)
    gt_summary = tf.summary.image("ground_truth", tf.reshape(gt_ph,
        [FLAGS.batch_size*FLAGS.sequence_length, features_dim[1],
            features_dim[2], 1]),
        max_outputs=FLAGS.batch_size*FLAGS.sequence_length)
    merged_summary = tf.summary.merge_all()

    with tf.Session() as sess:
        writer = tf.summary.FileWriter(real_log_dir, sess.graph)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        sess.run(tf.global_variables_initializer())
        if compressor and trained:
            saver.restore(sess, ckpt_path)

        with open('generated_sequences.pkl', 'r') as f:
            generated_sequences = pickle.load(f)
        generated_sequences = [x for x in generated_sequences if x[-2] >=
                FLAGS.sequence_length]
        frame_names, real_gts, index = default_get_batch(0,
                FLAGS.batch_size, FLAGS.sequence_length, generated_sequences)
        feed_dict = {file_names_placeholder:
                    frame_names}
        sess.run(enqueue_op, feed_dict=feed_dict)
        if pca:
            real_features = sess.run(features)
            """do pca"""
            print("reshaping")
            real_features = np.reshape(real_features, [
                FLAGS.batch_size*FLAGS.sequence_length*num_features,
                features_dim[-1]])
            print("doing pca")
            pca = PCA(n_components=FLAGS.compress_dim)
            real_features = pca.fit_transform(real_features)
            real_features = np.reshape(real_features, [
                FLAGS.batch_size, FLAGS.sequence_length,num_features,
                FLAGS.compress_dim])
            print("extracting similarity")
            simi, gt = sess.run([similarity_summary, gt_summary], feed_dict = {
                gt_ph: real_gts,
                pca_features: real_features
                })
            writer.add_summary(simi, 0)
            writer.add_summary(gt, 0)
        else:
            print(real_gts.shape, gt_ph.get_shape().as_list())
            summary = sess.run(merged_summary, feed_dict = {
                gt_ph: real_gts
                })
            writer.add_summary(summary, 0)
        sess.run(q_close_op) #close the queue
        coord.request_stop()
        coord.join(threads)

def copy_paste(width=3, length=FLAGS.sequence_length):
    """
    run a simple copy paste experiment
    with plain ntm tracker
    """
    total_length = 2*length+1
    input_ph = tf.placeholder(tf.float32, [FLAGS.batch_size*width*length])
    inputs = tf.reshape(input_ph, [FLAGS.batch_size, width, length])
    input_indicator_bit_pad = tf.zeros([FLAGS.batch_size, 1, length])
    inputs = tf.concat([inputs, input_indicator_bit_pad], 1)
    input_pad = tf.zeros_like(inputs)
    delimiter = tf.concat([
        tf.zeros([FLAGS.batch_size, width, 1]),
        tf.ones([FLAGS.batch_size, 1, 1])], 1)
    #input length is 2*length+1
    labels = tf.concat([input_pad, tf.zeros_like(delimiter), inputs], 2)
    tf.summary.image('labels', tf.reshape(labels, [FLAGS.batch_size,
        width+1, total_length, 1]), max_outputs=FLAGS.batch_size)
    #[batch, width+1, length]
    inputs = tf.concat([inputs, delimiter, input_pad], 2)
    tf.summary.image('inputs', tf.reshape(inputs, [FLAGS.batch_size,
        width+1, total_length, 1]), max_outputs=FLAGS.batch_size)
    initializer = tf.random_uniform_initializer(-FLAGS.init_scale,FLAGS.init_scale)
    ntm = LoopNTMTracker(total_length, width+1,
            initializer,
            mem_size=FLAGS.mem_size, mem_dim=FLAGS.mem_dim,
            controller_num_layers=FLAGS.num_layers,
            controller_hidden_size=FLAGS.hidden_size,
            read_head_size=FLAGS.read_head_size,
            write_head_size=FLAGS.write_head_size,
            write_first=FLAGS.write_first,)
    #ntm = PlainNTMTracker(total_length, width+1,
    #        initializer,
    #        mem_size=FLAGS.mem_size, mem_dim=FLAGS.mem_dim,
    #        controller_num_layers=FLAGS.num_layers,
    #        controller_hidden_size=FLAGS.hidden_size,
    #        read_head_size=FLAGS.read_head_size,
    #        write_head_size=FLAGS.write_head_size,
    #        write_first=FLAGS.write_first,)
    #input will be transposed to [batch, length, width+1]
    outputs, output_logits, Ms, ws, reads = ntm(tf.transpose(inputs,
        perm=[0,2,1]))
    #outputs, output_logits, states, debugs = ntm(tf.transpose(inputs,
    #    perm=[0,2,1]))
    """
    add summaries
    """
    tf.summary.image('M', tf.reshape(Ms,
        [FLAGS.batch_size, FLAGS.mem_size, FLAGS.mem_dim*total_length, 1]),
        max_outputs=FLAGS.batch_size)
    """w"""
    tf.summary.image('w_reads',
            tf.reshape(ws[:,:FLAGS.read_head_size,:,:],
                [FLAGS.batch_size,
                    FLAGS.mem_size*FLAGS.read_head_size, total_length, 1]),
            max_outputs=FLAGS.batch_size)
    tf.summary.image('w_writes',
            tf.reshape(ws[:,FLAGS.read_head_size:,:,:],
                [FLAGS.batch_size,
                    FLAGS.mem_size*FLAGS.write_head_size, total_length, 1]),
            max_outputs=FLAGS.batch_size)
    """reads"""
    tf.summary.image('reads',
            tf.reshape(reads, [FLAGS.batch_size*FLAGS.read_head_size,
                FLAGS.mem_dim, total_length, 1]),
            max_outputs=FLAGS.batch_size*FLAGS.read_head_size)

    output_sigmoid = tf.sigmoid(tf.transpose(output_logits, perm=[0,2,1]))
    tf.summary.image('output_sigmoid', tf.reshape(output_sigmoid, [FLAGS.batch_size,
        width+1, total_length, 1]), max_outputs=FLAGS.batch_size)
    loss_op = tf.losses.log_loss(labels, output_sigmoid)
    tf.summary.scalar('loss', loss_op)
    """training"""
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(loss_op, tvars),
            FLAGS.max_gradient_norm)
    optimizer = tf.train.RMSPropOptimizer(FLAGS.learning_rate,
            decay=FLAGS.decay, momentum=FLAGS.momentum)
    train_op = optimizer.apply_gradients(
            zip(grads, tvars),
            global_step = tf.contrib.framework.get_or_create_global_step())
    merged_summary = tf.summary.merge_all()
    #check_op = tf.add_check_numerics_ops()

    with tf.Session() as sess:
        print("session started")
        writer = tf.summary.FileWriter(real_log_dir, sess.graph)
        sess.run(tf.global_variables_initializer())
        """
        generate random inputs
        """
        for epoch in xrange(FLAGS.num_epochs):
            #print("running epoch {}".format(epoch))
            real_input = np.random.randint(2, size=FLAGS.batch_size*width*length)
            (loss, out, summ, _,
            #_
            ) = sess.run([loss_op, output_sigmoid,
                merged_summary, train_op,
                #check_op
                ],
                    feed_dict={
                        input_ph: real_input,
                        })
            print("{}: loss {}".format(epoch, loss))
            writer.add_summary(summ, epoch)

def ntm_sevenbyseven():
    """
    sequential means instead of presenting the whole feature map at once, I
    present each feature one by one
    1. create graph
    """
    """get the inputs"""
    train_summaries = []
    val_summaries = []
    file_names_placeholder, batch_img, batch_gt, q_close_op =\
            get_input(FLAGS.batch_size*FLAGS.sequence_length)
    train_summaries.append(tf.summary.image('train_batch_img', batch_img,
        max_outputs=FLAGS.batch_size*FLAGS.sequence_length))
    val_summaries.append(tf.summary.image('val_batch_img', batch_img,
        max_outputs=FLAGS.batch_size*FLAGS.sequence_length))
    """import VGG"""
    vgg_graph_def = tf.GraphDef()
    with open(FLAGS.vgg_model_frozen, "rb") as f:
        vgg_graph_def.ParseFromString(f.read())
    """the features"""
    features = tf.import_graph_def(vgg_graph_def, input_map={'inputs':
        batch_img}, return_elements=[FLAGS.feature_layer])[0]
    features_dim = features.get_shape().as_list()
    num_channels = features_dim[-1]
    print('features_dim', features_dim)
    num_features = features_dim[1]*features_dim[2]
    if FLAGS.compressor:
        """compress input dimensions"""
        w = tf.get_variable('input_compressor_w',
                shape=(1,1,features_dim[-1],FLAGS.compress_dim), dtype=tf.float32,
                initializer=tf.contrib.layers.xavier_initializer())
        features = tf.nn.conv2d(features, w, strides=(1,1,1,1), padding="VALID",
                name="input_compressor")
        num_channels = FLAGS.compress_dim
    print("num_channels:", num_channels)
    """
    the inputs;
    features is of shape [batch * seq_length, 28, 28, 128]
    originally it's reshaped to [batch, seq_len, num_features*num_channels]
    now we want it to be [batch, seq_len*num_features, 128]
    """

    inputs = tf.reshape(features, shape=[FLAGS.batch_size,
        FLAGS.sequence_length, num_features, num_channels], name="reshaped_inputs")
    """
    ground truth
    """
    gts = tf.reshape(batch_gt,
            [FLAGS.batch_size, FLAGS.sequence_length, num_features],
            name="ground_truth")
    #print('reshaped inputs:', inputs.get_shape())
    """
    placeholder to accept target indicator input
    because it's only for the 0th frame, so it's 2d
    """
    target = gts[:,0,:]
    """
    build the tracker inputs
    the inputs should be a matrix of [batch_size, xxx, 128+1+1]
    xxx:
        [0:num_features]: first frame, all features
        [num_features:num_features+1]: frame delimiter, [129]=1
        [num_features+1:num_features+2]: second frame, first feature
        [num_features+2:num_features+3]: second frame, feature delimiter, [128]=1
        [num_features+3:num_features+4]: second frame, second feature
        [num_features+4:num_features+5]: second frame, feature delimiter, [128]=1
        ...
    there will ultimately be
    num_features + (sequence_length - 1) * (1 + 2 * num_features) steps
    """
    total_steps = num_features + (FLAGS.sequence_length - 1) * (2 * num_features + 1)
    print("constructing inputs...")
    #shape [batch, seq_len, num_features, 130]
    inputs_padded = tf.concat([inputs, tf.zeros([FLAGS.batch_size,
        FLAGS.sequence_length, num_features, 2])], 3)
    #shape [batch, sequence_length-1, num_features, 130]
    inputs_no_zeroth = inputs_padded[:, 1:, :, :]
    #shape [batch, 1, 1, 128]
    dummy_feature = tf.zeros([FLAGS.batch_size, 1, 1, num_channels])
    #shape [batch, 1, 1, 130]
    frame_delimiter = tf.concat([
            dummy_feature,
            tf.zeros([FLAGS.batch_size, 1, 1, 1], dtype=tf.float32),
            tf.ones([FLAGS.batch_size, 1, 1, 1], dtype=tf.float32),
            ], 3)
    #frame delimiters, number: sequence_length - 1
    #shape [batch, sequence_length - 1, 1, 130]
    frame_delimiters = tf.tile(frame_delimiter,
            [1, FLAGS.sequence_length-1, 1, 1],
            name="frame_delimiters")
    feature_delimiter = tf.concat([
            dummy_feature,
            tf.ones([FLAGS.batch_size, 1, 1, 1], dtype=tf.float32),
            tf.zeros([FLAGS.batch_size, 1, 1, 1], dtype=tf.float32),
            ], 3)
    #feature delimiters, number: 1 per feature
    #shape [batch, sequence_length-1, num_features, 130]
    feature_delimiters = tf.tile(feature_delimiter,
            [1, FLAGS.sequence_length-1, num_features, 1],
            name="feature_delimiters")
    #now insert the feature delimiters
    inputs_no_zeroth = tf.reshape(tf.concat(
            [inputs_no_zeroth, feature_delimiters], 3),
            [FLAGS.batch_size, FLAGS.sequence_length-1, num_features*2,
                num_channels+2])
    #now insert the frame delimiters
    inputs_no_zeroth = tf.concat(
            [frame_delimiters, inputs_no_zeroth], 2)
    #now add back the zeroth frame
    inputs_no_zeroth = tf.reshape(inputs_no_zeroth,
            [
                FLAGS.batch_size,
                (FLAGS.sequence_length-1)*(2*num_features+1),
                num_channels+2])
    """
    num_features + (sequence_length - 1) * (1 + 2 * num_features) steps
    """
    inputs = tf.concat([
            inputs_padded[:,0,:,:],
            inputs_no_zeroth,
            ], 1, name="serial_inputs")
    target = tf.concat([
            target,
            tf.zeros([FLAGS.batch_size,
                (FLAGS.sequence_length - 1) * (2 * num_features + 1),
                ], dtype=tf.float32)], 1)
    #dims: [batch_size, total_steps, 131]
    inputs = tf.concat([
        inputs,
        tf.expand_dims(target, -1)], -1)
    print("constructing ground truths...")
    #tf.summary.image("ground_truth", tf.reshape(gt_ph,
    #    [-1,features_dim[1],features_dim[2],1]),
    #    max_outputs=FLAGS.batch_size*FLAGS.sequence_length)
    """
    there will be
    num_features + (sequence_length - 1) * (1 + 2 * num_features) steps

    how to produce?
    1. remove the first frame
    2. pad the features with num_features zeros
    3. pad the features with 1 zero at beginning
    4. pad at the beginning num_features zeros
    """

    """
    remove the first frame ground truth and create pad
    the dimension for gt_ph [batch_size, seq_length, num_features]
    """
    assert(FLAGS.sequence_length >= 2, "two_step must be used with sequence at least length 2")
    #gt_pad = tf.zeros_like(gts[:,1:,:], dtype=tf.float32, name="gt_pad")
    """
    stack at last axis, so that every feature scalar is prepended by a zero
    scalar
    """
    gt = gts[:,1:,:] #remove the first frame, used in loss calculation
    reshape_gts = tf.reshape(gts[:,1:,:],
        [FLAGS.batch_size*(FLAGS.sequence_length-1),features_dim[1],features_dim[2],1])
    train_summaries.append(tf.summary.image("train_ground_truth",
        reshape_gts,
        max_outputs=FLAGS.batch_size*(FLAGS.sequence_length-1)))
    val_summaries.append(tf.summary.image("val_ground_truth",
        reshape_gts,
        max_outputs=FLAGS.batch_size*(FLAGS.sequence_length-1)))
    #gt_stacked = tf.stack((gt_pad, gts[:,1:,:]), axis=3)
    #labels = tf.reshape(gt_stacked, [FLAGS.batch_size,
    #    FLAGS.sequence_length-1, 2*num_features])
    #"""
    #prepend each sequence with 1 zero, for the sequence delimiter
    #"""
    #labels = tf.concat([
    #    tf.zeros([FLAGS.batch_size, FLAGS.sequence_length-1, 1]),
    #    labels], 2)
    #labels = tf.reshape(labels,
    #        [FLAGS.batch_size, (FLAGS.sequence_length-1)*(2*num_features+1)])
    #"""
    #now prepend the ground truth for the zeroth frame
    #"""
    #first_frame_gt = tf.zeros([FLAGS.batch_size, num_features],
    #        name="gt_first_frame")
    #labels=tf.concat([first_frame_gt, labels], axis=1, name="labels")
    #tf.summary.image("labels", tf.reshape(labels,
    #    [1,FLAGS.batch_size,num_features+(FLAGS.sequence_length-1)*(2*num_features+1),1]),
    #    max_outputs=1)
    #labels = tf.expand_dims(labels, -1)

    print("constructing tracker...")
    """the tracker"""
    initializer = tf.random_uniform_initializer(-FLAGS.init_scale,FLAGS.init_scale)
    tracker = LoopNTMTracker(total_steps, 1,
            initializer,
            mem_size=FLAGS.mem_size, mem_dim=FLAGS.mem_dim,
            controller_num_layers=FLAGS.num_layers,
            controller_hidden_size=FLAGS.hidden_size,
            read_head_size=FLAGS.read_head_size,
            write_head_size=FLAGS.write_head_size,
            write_first=FLAGS.write_first,)
    """
    shape of outputs: [batch, model_length, 1]
    """
    print(inputs.get_shape().as_list())
    (outputs, output_logits,
            #Ms, ws, reads
            ) = tracker(inputs)
    print(output_logits.get_shape().as_list())
    #"""
    #add summaries
    #"""
    #print(Ms.get_shape().as_list())
    #reshape_Ms = tf.reshape(Ms,
    #    [FLAGS.batch_size, FLAGS.mem_size, FLAGS.mem_dim*total_steps, 1])
    #train_summaries.append(tf.summary.image('train_M', reshape_Ms,
    #    max_outputs=FLAGS.batch_size))
    #val_summaries.append(tf.summary.image('val_M', reshape_Ms,
    #    max_outputs=FLAGS.batch_size))
    #"""w"""
    #print(ws.get_shape().as_list())
    #reshape_w_reads = tf.reshape(ws[:,:FLAGS.read_head_size,:,:],
    #        [FLAGS.batch_size, FLAGS.mem_size*FLAGS.read_head_size, total_steps, 1])
    #reshape_w_writes = tf.reshape(ws[:,FLAGS.read_head_size:,:,:],
    #        [FLAGS.batch_size, FLAGS.mem_size*FLAGS.read_head_size, total_steps, 1])
    #train_summaries.append(tf.summary.image('train_w_reads', reshape_w_reads,
    #        max_outputs=FLAGS.batch_size))
    #train_summaries.append(tf.summary.image('train_w_writes', reshape_w_writes,
    #        max_outputs=FLAGS.batch_size))
    #val_summaries.append(tf.summary.image('val_w_reads', reshape_w_reads,
    #        max_outputs=FLAGS.batch_size))
    #val_summaries.append(tf.summary.image('val_w_writes', reshape_w_writes,
    #        max_outputs=FLAGS.batch_size))
    #"""reads"""
    #reshape_reads = tf.reshape(reads, [FLAGS.batch_size*FLAGS.read_head_size,
    #            FLAGS.mem_dim, total_steps, 1])
    #train_summaries.append(tf.summary.image('train_reads', reshape_reads,
    #        max_outputs=FLAGS.batch_size*FLAGS.read_head_size))
    #val_summaries.append(tf.summary.image('val_reads', reshape_reads,
    #        max_outputs=FLAGS.batch_size*FLAGS.read_head_size))

    """
    now the subgraph to convert model output sequence to perceivable heatmaps
    """
    output_gather = tf.squeeze(output_logits, axis=2)
    """remove the output for first frame"""
    output_gather = output_gather[:,num_features:]
    #output_first_frame = output_gather[:, :num_features]
    """remove the output for sequence delimiter"""
    output_gather = tf.reshape(output_gather, [FLAGS.batch_size,
        FLAGS.sequence_length-1, 2*num_features+1])
    #output_sequence_delimiter = output_gather[:,:,:1]
    output_gather = output_gather[:,:,1:]
    """remove the output of first step in 2-step presentation"""
    output_gather = tf.reshape(output_gather, [FLAGS.batch_size,
        FLAGS.sequence_length-1, num_features, 2])
    #output_first_step = output_gather[:,:,:,0]
    output_gather = output_gather[:,:,:,1]
    #other_outputs = tf.concat([
    #    tf.reshape(output_first_frame, [-1]),
    #    tf.reshape(output_sequence_delimiter, [-1]),
    #    tf.reshape(output_first_step, [-1])])
    output_sigmoids = tf.sigmoid(output_gather)
    reshape_output_gather = tf.reshape(output_sigmoids,
                [FLAGS.batch_size*(FLAGS.sequence_length-1),
                    features_dim[1],features_dim[2],1])
    train_summaries.append(tf.summary.image("train_gathered_outputs",
            reshape_output_gather,
            max_outputs=FLAGS.batch_size*(FLAGS.sequence_length-1)))
    val_summaries.append(tf.summary.image("val_gathered_outputs",
            reshape_output_gather,
            max_outputs=FLAGS.batch_size*(FLAGS.sequence_length-1)))
    #print('output_logits shape:', output_logits.get_shape())
    #output_logits is in [batch, seq_length, output_dim]
    #reshape it to [batch*seq_length, output_dim]
    print("constructing loss...")
    """soft max loss"""
    loss_op = tf.reduce_sum(
        tf.nn.softmax_cross_entropy_with_logits(
            logits=tf.reshape(output_gather, [-1, num_features]),
            labels=tf.reshape(gt, [-1, num_features])
            )) / (FLAGS.sequence_length-1)\
        #+ tf.losses.log_loss(tf.zeros_like(other_outputs), tf.sigmoid(other_outputs))
    """log loss"""
    #loss_op = tf.losses.log_loss(, tf.reshape(output_sigmoids, [
    #    FLAGS.batch_size*(FLAGS.sequence_length-1), num_features
    #    ]))
    train_summaries.append(tf.summary.scalar('train_loss', loss_op))
    val_loss_ph = tf.placeholder(tf.float32)
    val_loss_summary = tf.summary.scalar('val_loss', val_loss_ph)
    """training op"""
    tvars = tf.trainable_variables()
    # the gradient tensors
    global_step = tf.Variable(0, name='global_step', trainable=False)
    grads, _ = tf.clip_by_global_norm(tf.gradients(loss_op, tvars),
            FLAGS.max_gradient_norm)
    optimizer = tf.train.RMSPropOptimizer(FLAGS.learning_rate,
            decay=FLAGS.decay, momentum=FLAGS.momentum)
    train_op = optimizer.apply_gradients(
            zip(grads, tvars),
            global_step = global_step)
    train_merged_summary = tf.summary.merge(train_summaries)
    val_merged_summary = tf.summary.merge(val_summaries)

    """save training output images"""
    input_save = tf.cast(tf.reshape(batch_img+tf.expand_dims(VGG_MEAN,0),
        [FLAGS.batch_size, FLAGS.sequence_length, 224, 224, 3]), tf.uint8)
    gt_save = tf.reshape(batch_gt, [FLAGS.batch_size, FLAGS.sequence_length,
        FLAGS.gt_width, FLAGS.gt_width, 1])
    output_save = tf.reshape(output_gather, [FLAGS.batch_size,
        FLAGS.sequence_length-1, FLAGS.gt_width, FLAGS.gt_width, 1])
    output_save = tf.concat([tf.zeros_like(output_save[:,:1,:,:,:]),
        output_save], 1)

    return (#ops
            train_op, loss_op, q_close_op,
            #input placeholders
            file_names_placeholder, val_loss_ph,
            #summaries
            train_merged_summary,
            val_merged_summary,
            val_loss_summary,
            #tensor to plot to files,
            [input_save, gt_save, output_save],
            #global step variable
            global_step,
            sevenbyseven_get_batch)

def extract_features(input_feature_map, points=conv43Points):
    """
    input features is of dimension [batch, height, width, channels]
    """
    arr = []
    for y,x in points:
        arr.append(input_feature_map[:,y,x,:])
    return tf.stack(arr, axis=1, name="extracted_features"), len(points)



def ntm_8by8():
    """
    sequential means instead of presenting the whole feature map at once, I
    present each feature one by one
    1. create graph
    """
    """get the inputs"""
    train_summaries = []
    val_summaries = []
    file_names_placeholder, batch_img, batch_gt, q_close_op =\
            get_input(FLAGS.batch_size*FLAGS.sequence_length)
    train_summaries.append(tf.summary.image('train_batch_img', batch_img,
        max_outputs=FLAGS.batch_size*FLAGS.sequence_length))
    val_summaries.append(tf.summary.image('val_batch_img', batch_img,
        max_outputs=FLAGS.batch_size*FLAGS.sequence_length))
    """import VGG"""
    vgg_graph_def = tf.GraphDef()
    with open(FLAGS.vgg_model_frozen, "rb") as f:
        vgg_graph_def.ParseFromString(f.read())
    """the features"""
    features = tf.import_graph_def(vgg_graph_def, input_map={'inputs':
        batch_img}, return_elements=['vgg_16/conv4/conv4_3/Relu:0'])[0]
    features_dim = features.get_shape().as_list()
    features_dim[1] = features_dim[2] = 8
    features, num_features = extract_features(features)
    num_channels = features_dim[-1]
    print('features_dim', features_dim)
    print("num_channels:", num_channels)
    """
    the inputs;
    features is of shape [batch * seq_length, 28, 28, 128]
    originally it's reshaped to [batch, seq_len, num_features*num_channels]
    now we want it to be [batch, seq_len*num_features, 128]
    """

    inputs = tf.reshape(features, shape=[FLAGS.batch_size,
        FLAGS.sequence_length, num_features, num_channels], name="reshaped_inputs")
    print('inputs shape', inputs.get_shape().as_list())
    """
    ground truth
    """
    gts = tf.reshape(batch_gt,
            [FLAGS.batch_size, FLAGS.sequence_length, num_features],
            name="ground_truth")
    #print('reshaped inputs:', inputs.get_shape())
    """
    placeholder to accept target indicator input
    because it's only for the 0th frame, so it's 2d
    """
    target = gts[:,0,:]
    """
    build the tracker inputs
    the inputs should be a matrix of [batch_size, xxx, 128+1+1]
    xxx:
        [0:num_features]: first frame, all features
        [num_features:num_features+1]: frame delimiter, [129]=1
        [num_features+1:num_features+2]: second frame, first feature
        [num_features+2:num_features+3]: second frame, feature delimiter, [128]=1
        [num_features+3:num_features+4]: second frame, second feature
        [num_features+4:num_features+5]: second frame, feature delimiter, [128]=1
        ...
    there will ultimately be
    num_features + (sequence_length - 1) * (1 + 2 * num_features) steps
    """
    total_steps = num_features + (FLAGS.sequence_length - 1) * (2 * num_features + 1)
    print("constructing inputs...")
    #shape [batch, seq_len, num_features, 130]
    inputs_padded = tf.concat([inputs, tf.zeros([FLAGS.batch_size,
        FLAGS.sequence_length, num_features, 2])], 3)
    #shape [batch, sequence_length-1, num_features, 130]
    inputs_no_zeroth = inputs_padded[:, 1:, :, :]
    #shape [batch, 1, 1, 128]
    dummy_feature = tf.zeros([FLAGS.batch_size, 1, 1, num_channels])
    #shape [batch, 1, 1, 130]
    frame_delimiter = tf.concat([
            dummy_feature,
            tf.zeros([FLAGS.batch_size, 1, 1, 1], dtype=tf.float32),
            tf.ones([FLAGS.batch_size, 1, 1, 1], dtype=tf.float32),
            ], 3)
    #frame delimiters, number: sequence_length - 1
    #shape [batch, sequence_length - 1, 1, 130]
    frame_delimiters = tf.tile(frame_delimiter,
            [1, FLAGS.sequence_length-1, 1, 1],
            name="frame_delimiters")
    feature_delimiter = tf.concat([
            dummy_feature,
            tf.ones([FLAGS.batch_size, 1, 1, 1], dtype=tf.float32),
            tf.zeros([FLAGS.batch_size, 1, 1, 1], dtype=tf.float32),
            ], 3)
    #feature delimiters, number: 1 per feature
    #shape [batch, sequence_length-1, num_features, 130]
    feature_delimiters = tf.tile(feature_delimiter,
            [1, FLAGS.sequence_length-1, num_features, 1],
            name="feature_delimiters")
    #now insert the feature delimiters
    inputs_no_zeroth = tf.reshape(tf.concat(
            [inputs_no_zeroth, feature_delimiters], 3),
            [FLAGS.batch_size, FLAGS.sequence_length-1, num_features*2,
                num_channels+2])
    #now insert the frame delimiters
    inputs_no_zeroth = tf.concat(
            [frame_delimiters, inputs_no_zeroth], 2)
    #now add back the zeroth frame
    inputs_no_zeroth = tf.reshape(inputs_no_zeroth,
            [
                FLAGS.batch_size,
                (FLAGS.sequence_length-1)*(2*num_features+1),
                num_channels+2])
    """
    num_features + (sequence_length - 1) * (1 + 2 * num_features) steps
    """
    inputs = tf.concat([
            inputs_padded[:,0,:,:],
            inputs_no_zeroth,
            ], 1, name="serial_inputs")
    target = tf.concat([
            target,
            tf.zeros([FLAGS.batch_size,
                (FLAGS.sequence_length - 1) * (2 * num_features + 1),
                ], dtype=tf.float32)], 1)
    #dims: [batch_size, total_steps, 131]
    inputs = tf.concat([
        inputs,
        tf.expand_dims(target, -1)], -1)
    print("constructing ground truths...")
    #tf.summary.image("ground_truth", tf.reshape(gt_ph,
    #    [-1,features_dim[1],features_dim[2],1]),
    #    max_outputs=FLAGS.batch_size*FLAGS.sequence_length)
    """
    there will be
    num_features + (sequence_length - 1) * (1 + 2 * num_features) steps

    how to produce?
    1. remove the first frame
    2. pad the features with num_features zeros
    3. pad the features with 1 zero at beginning
    4. pad at the beginning num_features zeros
    """

    """
    remove the first frame ground truth and create pad
    the dimension for gt_ph [batch_size, seq_length, num_features]
    """
    assert(FLAGS.sequence_length >= 2, "two_step must be used with sequence at least length 2")
    #gt_pad = tf.zeros_like(gts[:,1:,:], dtype=tf.float32, name="gt_pad")
    """
    stack at last axis, so that every feature scalar is prepended by a zero
    scalar
    """
    gt = gts[:,1:,:] #remove the first frame, used in loss calculation
    reshape_gts = tf.reshape(gts[:,1:,:],
        [FLAGS.batch_size*(FLAGS.sequence_length-1),features_dim[1],features_dim[2],1])
    train_summaries.append(tf.summary.image("train_ground_truth",
        reshape_gts,
        max_outputs=FLAGS.batch_size*(FLAGS.sequence_length-1)))
    val_summaries.append(tf.summary.image("val_ground_truth",
        reshape_gts,
        max_outputs=FLAGS.batch_size*(FLAGS.sequence_length-1)))
    #gt_stacked = tf.stack((gt_pad, gts[:,1:,:]), axis=3)
    #labels = tf.reshape(gt_stacked, [FLAGS.batch_size,
    #    FLAGS.sequence_length-1, 2*num_features])
    #"""
    #prepend each sequence with 1 zero, for the sequence delimiter
    #"""
    #labels = tf.concat([
    #    tf.zeros([FLAGS.batch_size, FLAGS.sequence_length-1, 1]),
    #    labels], 2)
    #labels = tf.reshape(labels,
    #        [FLAGS.batch_size, (FLAGS.sequence_length-1)*(2*num_features+1)])
    #"""
    #now prepend the ground truth for the zeroth frame
    #"""
    #first_frame_gt = tf.zeros([FLAGS.batch_size, num_features],
    #        name="gt_first_frame")
    #labels=tf.concat([first_frame_gt, labels], axis=1, name="labels")
    #tf.summary.image("labels", tf.reshape(labels,
    #    [1,FLAGS.batch_size,num_features+(FLAGS.sequence_length-1)*(2*num_features+1),1]),
    #    max_outputs=1)
    #labels = tf.expand_dims(labels, -1)

    print("constructing tracker...")
    """the tracker"""
    initializer = tf.random_uniform_initializer(-FLAGS.init_scale,FLAGS.init_scale)
    tracker = LoopNTMTracker(total_steps, 1,
            initializer,
            mem_size=FLAGS.mem_size, mem_dim=FLAGS.mem_dim,
            controller_num_layers=FLAGS.num_layers,
            controller_hidden_size=FLAGS.hidden_size,
            read_head_size=FLAGS.read_head_size,
            write_head_size=FLAGS.write_head_size,
            write_first=FLAGS.write_first,)
    """
    shape of outputs: [batch, model_length, 1]
    """
    print(inputs.get_shape().as_list())
    (outputs, output_logits,
            #Ms, ws, reads
            ) = tracker(inputs)
    print(output_logits.get_shape().as_list())
    #"""
    #add summaries
    #"""
    #print(Ms.get_shape().as_list())
    #reshape_Ms = tf.reshape(Ms,
    #    [FLAGS.batch_size, FLAGS.mem_size, FLAGS.mem_dim*total_steps, 1])
    #train_summaries.append(tf.summary.image('train_M', reshape_Ms,
    #    max_outputs=FLAGS.batch_size))
    #val_summaries.append(tf.summary.image('val_M', reshape_Ms,
    #    max_outputs=FLAGS.batch_size))
    #"""w"""
    #print(ws.get_shape().as_list())
    #reshape_w_reads = tf.reshape(ws[:,:FLAGS.read_head_size,:,:],
    #        [FLAGS.batch_size, FLAGS.mem_size*FLAGS.read_head_size, total_steps, 1])
    #reshape_w_writes = tf.reshape(ws[:,FLAGS.read_head_size:,:,:],
    #        [FLAGS.batch_size, FLAGS.mem_size*FLAGS.read_head_size, total_steps, 1])
    #train_summaries.append(tf.summary.image('train_w_reads', reshape_w_reads,
    #        max_outputs=FLAGS.batch_size))
    #train_summaries.append(tf.summary.image('train_w_writes', reshape_w_writes,
    #        max_outputs=FLAGS.batch_size))
    #val_summaries.append(tf.summary.image('val_w_reads', reshape_w_reads,
    #        max_outputs=FLAGS.batch_size))
    #val_summaries.append(tf.summary.image('val_w_writes', reshape_w_writes,
    #        max_outputs=FLAGS.batch_size))
    #"""reads"""
    #reshape_reads = tf.reshape(reads, [FLAGS.batch_size*FLAGS.read_head_size,
    #            FLAGS.mem_dim, total_steps, 1])
    #train_summaries.append(tf.summary.image('train_reads', reshape_reads,
    #        max_outputs=FLAGS.batch_size*FLAGS.read_head_size))
    #val_summaries.append(tf.summary.image('val_reads', reshape_reads,
    #        max_outputs=FLAGS.batch_size*FLAGS.read_head_size))

    """
    now the subgraph to convert model output sequence to perceivable heatmaps
    """
    output_gather = tf.squeeze(output_logits, axis=2)
    """remove the output for first frame"""
    output_gather = output_gather[:,num_features:]
    #output_first_frame = output_gather[:, :num_features]
    """remove the output for sequence delimiter"""
    output_gather = tf.reshape(output_gather, [FLAGS.batch_size,
        FLAGS.sequence_length-1, 2*num_features+1])
    #output_sequence_delimiter = output_gather[:,:,:1]
    output_gather = output_gather[:,:,1:]
    """remove the output of first step in 2-step presentation"""
    output_gather = tf.reshape(output_gather, [FLAGS.batch_size,
        FLAGS.sequence_length-1, num_features, 2])
    #output_first_step = output_gather[:,:,:,0]
    output_gather = output_gather[:,:,:,1]
    #other_outputs = tf.concat([
    #    tf.reshape(output_first_frame, [-1]),
    #    tf.reshape(output_sequence_delimiter, [-1]),
    #    tf.reshape(output_first_step, [-1])])
    output_sigmoids = tf.sigmoid(output_gather)
    reshape_output_gather = tf.reshape(output_sigmoids,
                [FLAGS.batch_size*(FLAGS.sequence_length-1),
                    features_dim[1],features_dim[2],1])
    train_summaries.append(tf.summary.image("train_gathered_outputs",
            reshape_output_gather,
            max_outputs=FLAGS.batch_size*(FLAGS.sequence_length-1)))
    val_summaries.append(tf.summary.image("val_gathered_outputs",
            reshape_output_gather,
            max_outputs=FLAGS.batch_size*(FLAGS.sequence_length-1)))
    #print('output_logits shape:', output_logits.get_shape())
    #output_logits is in [batch, seq_length, output_dim]
    #reshape it to [batch*seq_length, output_dim]
    print("constructing loss...")
    """soft max loss"""
    loss_op = tf.reduce_sum(
        tf.nn.softmax_cross_entropy_with_logits(
            logits=tf.reshape(output_gather, [-1, num_features]),
            labels=tf.reshape(gt, [-1, num_features])
            )) / (FLAGS.sequence_length-1)\
        #+ tf.losses.log_loss(tf.zeros_like(other_outputs), tf.sigmoid(other_outputs))
    """log loss"""
    #loss_op = tf.losses.log_loss(, tf.reshape(output_sigmoids, [
    #    FLAGS.batch_size*(FLAGS.sequence_length-1), num_features
    #    ]))
    train_summaries.append(tf.summary.scalar('train_loss', loss_op))
    val_loss_ph = tf.placeholder(tf.float32)
    val_loss_summary = tf.summary.scalar('val_loss', val_loss_ph)
    """training op"""
    tvars = tf.trainable_variables()
    # the gradient tensors
    global_step = tf.Variable(0, name='global_step', trainable=False)
    grads, _ = tf.clip_by_global_norm(tf.gradients(loss_op, tvars),
            FLAGS.max_gradient_norm)
    optimizer = tf.train.RMSPropOptimizer(FLAGS.learning_rate,
            decay=FLAGS.decay, momentum=FLAGS.momentum)
    train_op = optimizer.apply_gradients(
            zip(grads, tvars),
            global_step = global_step)
    train_merged_summary = tf.summary.merge(train_summaries)
    val_merged_summary = tf.summary.merge(val_summaries)

    """save training output images"""
    input_save = tf.cast(tf.reshape(batch_img+tf.expand_dims(VGG_MEAN,0),
        [FLAGS.batch_size, FLAGS.sequence_length, 224, 224, 3]), tf.uint8)
    gt_save = tf.reshape(batch_gt, [FLAGS.batch_size, FLAGS.sequence_length,
        FLAGS.gt_width, FLAGS.gt_width, 1])
    output_save = tf.reshape(output_gather, [FLAGS.batch_size,
        FLAGS.sequence_length-1, FLAGS.gt_width, FLAGS.gt_width, 1])
    output_save = tf.concat([tf.zeros_like(output_save[:,:1,:,:,:]),
        output_save], 1)

    return (#ops
            train_op, loss_op, q_close_op,
            #input placeholders
            file_names_placeholder, val_loss_ph,
            #summaries
            train_merged_summary,
            val_merged_summary,
            val_loss_summary,
            #tensor to plot to files,
            [input_save, gt_save, output_save],
            #global step variable
            global_step,
            sevenbyseven_get_batch)

def find_validation_batch(target_step=1700):
    print("getting valid sequences...")
    _, train_seqs, val_seqs = get_valid_sequences()
    print('{} sequences after length filtering'.format(
        len(train_seqs)+len(val_seqs)))
    num_train = len(train_seqs)/FLAGS.batch_size*FLAGS.batch_size
    num_val = len(val_seqs)/FLAGS.batch_size*FLAGS.batch_size
    train_seqs = train_seqs[:num_train]
    val_seqs = val_seqs[:num_val]
    print('{} train seqs, {} val seqs'.format(
        len(train_seqs), len(val_seqs)))
    step = 0 #this is not global step, and is only relevant to logging
    random.shuffle(train_seqs)
    while True:
        if step % FLAGS.validation_interval == 0:
            random.shuffle(val_seqs)
        if step == target_step: break
        step += 1
    print(val_seqs)
    with open('validation_seqs_{}.pkl'.format(target_step), 'w') as f:
        pickle.dump(val_seqs, f)
    return val_seqs

def main(_):
    """
    1. create graph
    2. train and eval
    """
    if FLAGS.two_step:
        train_op, loss_op, merged_summary, target_ph, gt_ph,\
                file_names_placeholder, enqueue_op, q_close_op,\
                other_ops, get_batch = ntm_two_step()

        train_and_val(train_op, loss_op, merged_summary, target_ph, gt_ph,
                file_names_placeholder, enqueue_op, q_close_op, other_ops, get_batch)
    elif FLAGS.sequential:
        params = ntm_sequential()
        train_and_val_sequential(*params)
    elif FLAGS.sevenbyseven:
        if not FLAGS.sequences_dir:
            raise Exception('must provide FLAGS.sequences_dir')
        params = ntm_sevenbyseven()
        train_and_val_sevenbyseven(*params)
    elif FLAGS.eightbyeight:
        if not FLAGS.sequences_dir:
            raise Exception('must provide FLAGS.sequences_dir')
        params = ntm_8by8()
        train_and_val_sevenbyseven(*params)
    else:
        train_op, loss_op, merged_summary, target_ph, gt_ph,\
                file_names_placeholder, enqueue_op, q_close_op,\
                other_ops, get_batch = ntm()

        train_and_val(train_op, loss_op, merged_summary, target_ph, gt_ph,
                file_names_placeholder, enqueue_op, q_close_op, other_ops, get_batch)

if __name__ == '__main__':
    with open('visualize.sh', 'w') as f:
        f.write('tensorboard --logdir="{}"'.format(real_log_dir))
    if (FLAGS.test_read_imgs):
        test_read_imgs()
    elif FLAGS.test_input:
        test_get_input()
    elif (FLAGS.lstm_only):
        lstm_only()
    elif (FLAGS.sanity_check):
        sanity_check_compressor()
    elif (FLAGS.sanity_check_compressor):
        sanity_check_compressor(compressor=True)
    elif (FLAGS.sanity_check_trained_compressor):
        sanity_check_compressor(compressor=True, trained=True)
    elif (FLAGS.sanity_check_pca):
        sanity_check_compressor(pca=True)
    elif (FLAGS.copy_paste):
        copy_paste()
    elif (FLAGS.find_validation_batch):
        find_validation_batch()
    else:
        tf.app.run()
