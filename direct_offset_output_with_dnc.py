from __future__ import absolute_import

import tensorflow as tf
import numpy as np
from datetime import datetime
import os
import scipy.misc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import dnc

from receptive_field_sizes import conv43Points

import random

flags = tf.app.flags
#############
#model params
#############
flags.DEFINE_integer("mem_size", 128, "size of mem")
flags.DEFINE_integer("mem_dim", 64, "dim of mem")
flags.DEFINE_integer("hidden_size", 200, "number of LSTM cells")
flags.DEFINE_integer("num_layers", 1, "number of LSTM cells")
flags.DEFINE_integer("read_head_size", 4, "number of read heads")
flags.DEFINE_integer("write_head_size", 1, "number of write heads")
flags.DEFINE_boolean("reverse_image", False, "reverse horizontally the input image")

flags.DEFINE_integer("num_epochs", 1, "number of epochs to train")
flags.DEFINE_string("vgg_model_frozen", "./vgg_16_frozen.pb", "The pb file of the frozen vgg_16 network")
flags.DEFINE_string("log_dir", "./log", "The log dir")
flags.DEFINE_integer("sequence_length", 20, "The length of input sequences")
flags.DEFINE_integer("batch_size", 16, "size of batch")
flags.DEFINE_string("feature_layer", "vgg_16/conv4/conv4_3/Relu:0", "The layer of feature to be put into NTM as input")
flags.DEFINE_integer("max_gradient_norm", 50, "for gradient clipping normalization")
flags.DEFINE_float("learning_rate", 1e-4, "learning rate")
flags.DEFINE_float("momentum", 0.9, "learning rate")
flags.DEFINE_float("decay", 0.95, "learning rate")
flags.DEFINE_string("tag", "", "tag for the log record")
flags.DEFINE_string("ckpt_path", "", "path for the ckpt file to be restored")
flags.DEFINE_integer("log_interval", 10, "number of epochs before log")
flags.DEFINE_float("init_scale", 0.05, "initial range for weights")
flags.DEFINE_boolean("test_input", False, "test the new get_input function")
flags.DEFINE_integer("gt_width", 8, "width of ground truth. a value of 7 means a 7x7 ground truth")
flags.DEFINE_integer("gt_depth", 8, "number of bytes used for each pixel")
flags.DEFINE_string("sequences_dir", "", "dir to look for sequences")
flags.DEFINE_integer("validation_interval", 100, "number of steps before validation")
flags.DEFINE_integer("validation_batch", 1, "validate only this number of batches")

FLAGS = flags.FLAGS

random.seed(42)

real_log_dir = os.path.abspath(os.path.join(FLAGS.log_dir,
    str(datetime.now())+FLAGS.tag))
print('real log dir: {}'.format(real_log_dir))

VGG_MEAN = tf.constant([123.68, 116.78, 103.94], dtype=tf.float32,
        shape=[1,1,3], name="VGG_MEAN")

def run_model(input_sequence, output_size):
  """Runs model on input sequence."""

  access_config = {
      "memory_size": FLAGS.mem_size,
      "word_size": FLAGS.mem_dim,
      "num_reads": FLAGS.read_head_size,
      "num_writes": FLAGS.write_head_size,
  }
  controller_config = {
      "hidden_size": FLAGS.hidden_size,
  }
  clip_value = FLAGS.max_gradient_norm

  dnc_core = dnc.DNC(access_config, controller_config, output_size, clip_value)
  initial_state = dnc_core.initial_state(FLAGS.batch_size)
  output_sequence, _ = tf.nn.dynamic_rnn(
      cell=dnc_core,
      inputs=input_sequence,
      time_major=True,
      initial_state=initial_state)

  return output_sequence

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
                #print(img.shape)
                if len(img.shape) > 3:
                    ax.imshow(np.squeeze(img[batch_idx, length_idx,:,:,:]))
                    ax.axis('off')
                else:
                    #print(img[batch_idx, length_idx])
                    ax.set_xlim(-.5,.5)
                    ax.set_ylim(-.5,.5)
                    ax.plot([img[batch_idx, length_idx, 1]],
                            [-img[batch_idx, length_idx, 0]],
                            marker='o', markersize=3,
                            color="red")
                    #xy = (img[batch_idx, length_idx, 1],
                    #      img[batch_idx, length_idx, 0])
                    #ax.annotate('(%s, %s)' % xy, xy=xy, textcoords='data')
    fig.savefig(os.path.join(savedir, filename+'.png'),
            bbox_inches='tight', pad_inches=0)
    plt.close(fig)

def get_valid_sequences(sequences_dir=FLAGS.sequences_dir, min_length=FLAGS.sequence_length):
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
        skip = len(files) / min_length
        if skip == 0:
            continue
        sliced = files[::skip][:min_length]
        result.append((seqdir, sliced))
        if 'train' in seqdir:
            train.append((seqdir, sliced))
        elif 'val' in seqdir:
            val.append((seqdir, sliced))
        else:
            raise Exception('expect either train or val in sequence name')
    return result, train, val

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
    record_defaults = [[.0],[.0],[.0],[.0],[.0],[.0],[.0],[.0],[''],[.0],[.0]]
    y1,x1,y2,x2,_,_,_,_,img_filename,y_offset,x_offset = tf.decode_csv(value, record_defaults)
    cropbox = tf.stack([y1,x1,y2,x2])
    cropboxes, img_filenames, y_offsets, x_offsets = tf.train.batch(
            [cropbox, img_filename, y_offset, x_offset],
            batch_size = batch_size, num_threads=1)
    if FLAGS.reverse_image:
        x_offsets = -x_offsets
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
    print(batch_img.get_shape().as_list())
    if FLAGS.reverse_image:
        batch_img = tf.reverse(batch_img, [2])
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

    return filename_nosuffix_ph, batch_img, batch_gt, y_offsets, x_offsets, close_qs_op


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

def extract_features(input_feature_map, points=conv43Points):
    """
    input features is of dimension [batch, height, width, channels]
    """
    arr = []
    for y,x in points:
        arr.append(input_feature_map[:,y,x,:])
    return tf.stack(arr, axis=1, name="extracted_features"), len(points)

def ntm_offsets():
    """
    sequential means instead of presenting the whole feature map at once, I
    present each feature one by one
    1. create graph
    """
    """get the inputs"""
    train_summaries = []
    val_summaries = []
    file_names_placeholder, batch_img, batch_gt, y_offsets, x_offsets, q_close_op =\
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
    """
    extract an 8 by 8 sub map from the conv4_3 feature map
    """
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
            name="gt_heatmap")
    #print('reshaped inputs:', inputs.get_shape())
    """
    placeholder to accept target indicator input
    because it's only for the 0th frame, so it's 2d
    """
    target = gts[:,0,:]
    """
    build the tracker inputs
    the inputs should be a matrix of [batch_size, total_steps, num_features, feature_depth+2]
    feature_depth+1-th bit is target indicator
    feature_depth-th bit is frame delimiter
    """
    total_steps = FLAGS.sequence_length * (num_features + 1)
    print("constructing inputs...")
    #shape [batch, seq_len, num_features, depth+1]
    inputs_padded = tf.concat([inputs, tf.zeros([FLAGS.batch_size,
        FLAGS.sequence_length, num_features, 1])], 3)
    #shape [batch, sequence_length-1, num_features, depth+1]
    #shape [1, 1, 1, depth]
    dummy_feature = tf.zeros([1, 1, 1, num_channels])
    #shape [1, 1, 1, depth+1]
    frame_delimiter = tf.concat([
            dummy_feature,
            tf.ones([1, 1, 1, 1], dtype=tf.float32),
            ], 3)
    #frame delimiters, number: sequence_length
    #shape [batch, sequence_length - 1, 1, 130]
    frame_delimiters = tf.tile(frame_delimiter,
            [FLAGS.batch_size, FLAGS.sequence_length, 1, 1],
            name="frame_delimiters")
    #now insert the frame delimiters
    #NOTE: in this case, we put the delimiters at the end of each frame
    inputs_padded = tf.concat(
            [inputs_padded, frame_delimiters], 2)
    #now add back the zeroth frame
    inputs_padded = tf.reshape(inputs_padded,
            [
                FLAGS.batch_size,
                FLAGS.sequence_length*(num_features+1),
                num_channels+1
            ])
    #now add the target indicators
    #put zero for this row on every subsequent frame
    #[batch_size, seq_len*(features+1)]
    target = tf.concat([
            target,
            tf.zeros([FLAGS.batch_size,
                (FLAGS.sequence_length - 1) * (num_features + 1)+1,
                ], dtype=tf.float32)], 1)
    #dims: [batch_size, total_steps, 131]
    inputs = tf.concat([
        inputs_padded,
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
    offsets = tf.stack([y_offsets, x_offsets], axis=1)
    offsets = tf.reshape(offsets, [FLAGS.batch_size, FLAGS.sequence_length, 2])
    #reshape_gts = tf.reshape(gts[:,1:,:],
    #    [FLAGS.batch_size*(FLAGS.sequence_length-1),features_dim[1],features_dim[2],1])
    #train_summaries.append(tf.summary.image("train_ground_truth",
    #    reshape_gts,
    #    max_outputs=FLAGS.batch_size*(FLAGS.sequence_length-1)))
    #val_summaries.append(tf.summary.image("val_ground_truth",
    #    reshape_gts,
    #    max_outputs=FLAGS.batch_size*(FLAGS.sequence_length-1)))

    print("constructing tracker...")
    """the tracker"""
    input_sequence = tf.transpose(inputs, perm=[1,0,2])
    output_sequence = run_model(input_sequence, 2)
    """
    convert time-major into batch-major
    """
    output_logits = tf.transpose(output_sequence, perm=[1,0,2])
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
    """remove the output for first frame"""
    output_gather = output_logits[:,num_features+1:,:]
    #output_first_frame = output_gather[:, :num_features]
    """extract the output for frame delimiter"""
    output_gather = tf.reshape(output_gather, [FLAGS.batch_size,
        FLAGS.sequence_length-1, num_features+1, 2])
    #output_sequence_delimiter = output_gather[:,:,:1]
    #shape of output_gather: [batch, sequence length-1, 2]
    output_gather = output_gather[:,:,num_features,:]
    #other_outputs = tf.concat([
    #    tf.reshape(output_first_frame, [-1]),
    #    tf.reshape(output_sequence_delimiter, [-1]),
    #    tf.reshape(output_first_step, [-1])])
    output_sigmoids = tf.tanh(output_gather)
    #print('output_logits shape:', output_logits.get_shape())
    #output_logits is in [batch, seq_length, output_dim]
    #reshape it to [batch*seq_length, output_dim]
    print("constructing loss...")
    #"""soft max loss"""
    #loss_op = tf.reduce_sum(
    #    tf.nn.softmax_cross_entropy_with_logits(
    #        logits=tf.reshape(output_gather, [-1, num_features]),
    #        labels=tf.reshape(gt, [-1, num_features])
    #        )) / (FLAGS.sequence_length-1)\
    #    #+ tf.losses.log_loss(tf.zeros_like(other_outputs), tf.sigmoid(other_outputs))
    """log loss"""
    loss_op = tf.nn.l2_loss(output_sigmoids-offsets[:,1:,:])
    train_summaries.append(tf.summary.scalar('train_loss', loss_op))
    val_loss_ph = tf.placeholder(tf.float32)
    val_loss_summary = tf.summary.scalar('val_loss', val_loss_ph)
    """training op"""
    tvars = tf.trainable_variables()
    # the gradient tensors
    global_step = tf.get_variable(
        name="global_step",
        shape=[],
        dtype=tf.int64,
        initializer=tf.zeros_initializer(),
        trainable=False,
        collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.GLOBAL_STEP])
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
    gt_save = tf.reshape(offsets, [FLAGS.batch_size, FLAGS.sequence_length,
        2])
    output_save = tf.reshape(output_gather, [FLAGS.batch_size,
        FLAGS.sequence_length-1, 2])
    #restore the length of zeroth frame
    output_save = tf.concat([tf.zeros_like(output_save[:,:1,:]),
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

def main(_):
    """
    1. create graph
    2. train and eval
    """
    tf.logging.set_verbosity(3)  # Print INFO log messages.
    if not FLAGS.sequences_dir:
        raise Exception('must provide FLAGS.sequences_dir')
    params = ntm_offsets()
    train_and_val_sevenbyseven(*params)

if __name__ == '__main__':
    with open('visualize.sh', 'w') as f:
        f.write('tensorboard --logdir="{}"'.format(real_log_dir))
    with open('test_tracker.sh', 'w') as f:
        f.write('python test_tracker --ckpt_path="{}/model.ckpt-xxxx"'.format(real_log_dir))
    if FLAGS.test_input:
        test_get_input()
    else:
        tf.app.run()
