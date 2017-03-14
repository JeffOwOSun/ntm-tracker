"""
interface for the VOTchallenge
"""
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
flags.DEFINE_string("ckpt_path", "", "path for the ckpt file to be restored")

FLAGS=flags.FLAGS

VGG_MEAN = tf.constant([123.68, 116.78, 103.94], dtype=tf.float32,
        shape=[1,1,3], name="VGG_MEAN")

def softmax(x):
    e_x = np.exp(x)
    return e_x / e_x.sum()

class NTMTracker(object):
    def __init__(self, imagepath, region):
        """
        1. create an NTMTracker
        2. initialize with the image and region
        """
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
        outputs = softmax(self._collect_outputs(outputs))
        #TODO: convert this outputs to bbox
        return self._get_bbox(outputs)

    def _get_bbox(outputs):
        pass

    def _collect_outputs(self, outputs):
        """
        length of outputs is 2*num_features+1
        """
        #remove first output that corresponds with frame delimiter
        #then take the second of every two items
        outputs = outputs[2::2]
        assert(len(outputs) == self.num_features)
        return np.concat(outputs)
        #locate the bbox

    def _run_tracker(self, sess, inputs):
        length, depth = inputs.shape
        outputs = []
        for idx in xrange(length):
            output_logit, state = sess.run(
                    [self.output_logit, self.state],
                    feed_dict = {
                        self.state_ph: self.states[-1],
                        self.input_ph: inputs[idx],
                        })
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
        self.cropbox = preprocess.calculate_cropbox(self.normalized_bbox)
        #transformation to map cropbox to [0,0,1,1]
        self.transformation = preprocess.calculate_transformation(self.cropbox)

    def _build_tracker(self, **kwargs):
        self.input_ph = tf.placeholder(shape=(FLAGS.input_depth),
                dtype=tf.float32)
        self.cell = NTMCell(1, **kwargs)
        with tf.variable_scope('ntm-tracker'):
            self.zero_state = self.cell.zero_state(1)
            self.state_ph = self.cell.state_placeholder(1)
            output, self.output_logit, self.state, _, _, _, _, _ = self.cell(self.input_ph, self.state_ph)

    def _build_preprocessor(self, crop_size=[224,224]):
        """
        build the graph for preprocessor, run only once
        """
        self.image_ph = tf.placeholder(tf.float32)
        self.cropbox_ph = tf.placeholder(tf.float32, shape=[4]) #y1,x1,y2,x2
        img = self.image_ph - VGG_MEAN
        batch_img = tf.crop_and_resize(
                tf.expand_dims(img, 0),
                tf.expand_dims(self.bbox_ph, 0),
                [0], crop_size)
        vgg_graph_def = tf.GraphDef()
        with open(FLAGS.vgg_model_frozen, "rb") as f:
            vgg_graph_def.ParseFromString(f.read())
        features = tf.import_graph_def(
            vgg_graph_def, input_map={'inputs': batch_img},
            return_elements=[FLAGS.feature_layer])[0]
        features_dim = features.get_shape().as_list()
        self.features_dim = features_dim
        print('features_dim', features_dim)
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
        num_features = feature_dim[0]*feature_dim[1]
        num_channels = feature_dim[2]
        if is_first_frame:
            #generate gt
            gt = np.reshape(preprocess.generate_gt(
                    preprocess.apply_transformation(self.normalized_bbox,
                            self.transformation),
                    FLAGS.cropbox_grid, FLAGS.bbox_grid), [-1, 1])
            features = np.concat([features, gt], 1)
            pad = np.zeros((gt.shape[0], 2))
            features = np.concat([features, pad], 1)
        else:
            pad = np.zeros((num_features, 3))
            features = np.concat([features, pad], 1)
            #[...,0,0,1]
            frame_delimiter = np.concat([np.zeros((1, num_channels+2)),
                np.ones((1,1))], 1)
            #[...,0,1,0]
            feature_delimiters = np.concat([np.zeros((num_features, num_channels+1)),
                np.ones((num_features,1)), np.zeros((num_features,1))], 1)
            features = np.reshape(np.stack([features, feature_delimiters], axis=-1),
                    [num_features*2, num_channels+3])
            features = np.concat([frame_delimiter, features], 0)
        return features

