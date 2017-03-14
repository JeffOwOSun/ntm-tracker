"""
interface for the VOTchallenge
"""
import sys
import vot
from ntm_cell import NTMCell
import preprocess

import tensorflow as tf
import numpy as np
import scipy.misc

flags = tf.app.flags
flags.DEFINE_integer("input_depth", 515, "depth of input feature")
flags.DEFINE_string("vgg_model_frozen", "./vgg_16_frozen.pb", "The pb file of the frozen vgg_16 network")
flags.DEFINE_string("feature_layer", "vgg_16/pool5/MaxPool:0", "The layer of feature to be put into NTM as input")
flags.DEFINE_integer("cropbox_grid", 7, "side length of grid, on which the ground truth will be generated")
flags.DEFINE_integer("bbox_grid", 3, "side length of bbox grid")
flags.DEFINE_string("ckpt_path", "/home/jowos/git/ntm-tracker/log/2017-03-14 20:46:10.222973batch4-seqlen20-numlayer1-hidden500-epoch10-lr1e-4-rw1-7x7-write1st-saveimgs-skipframe-memdim20-memdize128-unifiedzerostate/model.ckpt-100", "path for the ckpt file to be restored")

FLAGS=flags.FLAGS

VGG_MEAN = tf.constant([123.68, 116.78, 103.94], dtype=tf.float32,
        shape=[1,1,3], name="VGG_MEAN")

print(np)
def softmax(x):
    e_x = np.exp(x)
    return e_x / e_x.sum()

class NTMTracker(object):
    def __init__(self, imagepath, region):
        """
        1. create an NTMTracker
        2. initialize with the image and region
        """
        self.init_region = region
        """
        create the ntm tracker
        1. prepare the input tensor
        """
        self._build_preprocessor()
        """
        2. build the tracker
        """
        self._build_tracker(mem_size=128, mem_dim=20, controller_num_layers=1,
                controller_hidden_size=500, read_head_size=1, write_head_size=1,
                write_first=True)
        #open the image
        image = scipy.misc.imread(imagepath)
        width, height, _ = image.shape
        self.image_size = (width, height)
        #update bbox
        self._update_bbox(self.image_size, region)
        #start the session
        self.sess = tf.Session()
        saver = tf.train.Saver()
        saver.restore(self.sess, FLAGS.ckpt_path)
        #preprocess the image
        features = self._preprocess_image(self.sess, image, True)
        #initialize the states with real zero_state value
        self.states = [self.sess.run(self.zero_state)]
        #run the tracker
        #this output is discarded
        self._run_tracker(self.sess, features)

    def track(self, imagepath):
        #open the image
        image = scipy.misc.imread(imagepath)
        features = self._preprocess_image(self.sess, image, False)
        outputs = self._run_tracker(self.sess, features)
        #print('np', np)
        collected = self._collect_outputs(outputs)
        #print('collected', collected)
        outputs = softmax(collected)
        #TODO: convert this outputs to bbox
        if not hasattr(self, 'frame'):
            self.frame = 0
        self.frame += 1
        scipy.misc.imsave('output_{}.png'.format(self.frame),
                np.reshape(outputs, [FLAGS.cropbox_grid,
                    FLAGS.cropbox_grid]))
        return self._get_bbox(outputs)

    def _get_bbox(self, outputs):
        return self.init_region

    def _collect_outputs(self, outputs):
        """
        length of outputs is 2*num_features+1
        """
        #remove first output that corresponds with frame delimiter
        #then take the second of every two items
        outputs = outputs[2::2]
        assert(len(outputs) == self.num_features)
        return np.concatenate(outputs, axis=0)
        #locate the bbox

    def _run_tracker(self, sess, inputs):
        length, depth = inputs.shape
        outputs = []
        for idx in xrange(length):
            feed_dict = {
                    self.input_ph: inputs[idx],
                    }
            for k, v in self.state_ph.items():
                feed_dict[v] = self.states[-1][k]

            output_logit, state = sess.run(
                    [self.output_logit, self.state],
                    feed_dict = feed_dict)
            outputs.append(output_logit)
            self.states.append(state)
        return outputs

    def _update_bbox(self, image_size, region):
        """
        the region is true region
        need to normalize the bbox
        """
        x1,y1,w,h = region
        bbox = (y1,x1,y1+h,x1+w)
        width, height = image_size
        #normalized bbox
        self.normalized_bbox = preprocess.normalize_bbox((width, height), bbox)
        #cropbox
        self.cropbox = preprocess.calculate_cropbox(self.normalized_bbox,
                FLAGS.cropbox_grid, FLAGS.bbox_grid)
        #transformation to map cropbox to [0,0,1,1]
        self.transformation = preprocess.calculate_transformation(self.cropbox)

    def _build_tracker(self, **kwargs):
        self.input_ph = tf.placeholder(shape=(FLAGS.input_depth),
                dtype=tf.float32)
        self.cell = NTMCell(1, **kwargs)
        with tf.variable_scope('ntm-tracker'):
            self.zero_state = self.cell.zero_state(1)
            self.state_ph = self.cell.state_placeholder(1)
            output, self.output_logit, self.state, _, _, _, _, _ =\
            self.cell(tf.expand_dims(self.input_ph, 0), self.state_ph)

    def _build_preprocessor(self, crop_size=[224,224]):
        """
        build the graph for preprocessor, run only once
        """
        self.image_ph = tf.placeholder(tf.float32)
        self.cropbox_ph = tf.placeholder(tf.float32, shape=[4]) #y1,x1,y2,x2
        img = self.image_ph - VGG_MEAN
        batch_img = tf.image.crop_and_resize(
                tf.expand_dims(img, 0),
                tf.expand_dims(self.cropbox_ph, 0),
                [0], crop_size)
        vgg_graph_def = tf.GraphDef()
        with open(FLAGS.vgg_model_frozen, "rb") as f:
            vgg_graph_def.ParseFromString(f.read())
        features = tf.import_graph_def(
            vgg_graph_def, input_map={'inputs': batch_img},
            return_elements=[FLAGS.feature_layer])[0]
        features_dim = features.get_shape().as_list()
        self.features_dim = features_dim
        #print('features_dim', features_dim)
        num_features = features_dim[1]*features_dim[2]
        self.num_features = num_features
        self.features = tf.reshape(features, [num_features, features_dim[3]])

    def _preprocess_image(self, sess, image, is_first_frame):
        features = sess.run(self.features, feed_dict = {
            self.image_ph: image,
            self.cropbox_ph: self.cropbox,
            })
        #by default should be [7, 7, 512]
        feature_dim = features.shape
        #print(feature_dim)
        num_features, num_channels = feature_dim
        if is_first_frame:
            #generate gt
            gt = np.reshape(preprocess.generate_gt(
                    preprocess.apply_transformation(self.normalized_bbox,
                            self.transformation),
                    FLAGS.cropbox_grid, FLAGS.bbox_grid), [-1, 1])
            features = np.concatenate([features, gt], 1)
            pad = np.zeros((gt.shape[0], 2))
            features = np.concatenate([features, pad], 1)
        else:
            pad = np.zeros((num_features, 3))
            features = np.concatenate([features, pad], 1)
            #[...,0,0,1]
            frame_delimiter = np.concatenate([np.zeros((1, num_channels+2)),
                np.ones((1,1))], 1)
            #[...,0,1,0]
            feature_delimiters = np.concatenate([np.zeros((num_features, num_channels+1)),
                np.ones((num_features,1)), np.zeros((num_features,1))], 1)
            features = np.reshape(np.stack([features, feature_delimiters], axis=-1),
                    [num_features*2, num_channels+3])
            features = np.concatenate([frame_delimiter, features], 0)
        return features

handle = vot.VOT("rectangle")
selection = handle.region()

imagefile = handle.frame()
if not imagefile:
    sys.exit(0)

print(selection, imagefile)

tracker = NTMTracker(imagefile, selection)
#import pdb; pdb.set_trace()
while True:
    imagefile = handle.frame()
    if not imagefile:
        break
    region = tracker.track(imagefile)
    handle.report(region)
