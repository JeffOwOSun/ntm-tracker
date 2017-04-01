"""
interface for the VOTchallenge
"""
import os
#disable GPU
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys
import vot
from ntm_cell import NTMCell
import preprocess

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as pts
from PIL import Image, ImageDraw

import tensorflow as tf
import numpy as np
import scipy.misc
import scipy.signal
from skimage import measure

flags = tf.app.flags
flags.DEFINE_integer("input_depth", 515, "depth of input feature")
flags.DEFINE_string("vgg_model_frozen", "/home/jowos/git/ntm-tracker/vgg_16_frozen.pb", "The pb file of the frozen vgg_16 network")
flags.DEFINE_string("feature_layer", "vgg_16/pool5/MaxPool:0", "The layer of feature to be put into NTM as input")
flags.DEFINE_integer("cropbox_grid", 7, "side length of grid, on which the ground truth will be generated")
flags.DEFINE_integer("bbox_grid", 3, "side length of bbox grid")
flags.DEFINE_string("ckpt_path", "/home/jowos/git/ntm-tracker/log/2017-03-18 02:58:32.305332batch4-seqlen20-numlayer1-hidden500-epoch10-lr1e-4-rw1-7x7-saveimgs-skipframe5-memdim20-memdize128-unifiedzerostate/model.ckpt-600", "path for the ckpt file to be restored")
flags.DEFINE_boolean("save_img", False, "whether to save intermediate outputs")

FLAGS=flags.FLAGS


def bb_iou(boxA, boxB):
    """
    each boxA,boxB is of format [x1, y1, x2, y2]
    """
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = (xB - xA + 1) * (yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

def softmax(x):
    e_x = np.exp(x)
    return e_x / e_x.sum()

def reject_outliers(data, m = 2.):
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d/mdev if mdev else 0.
    return data[s<m]

class NTMTracker(object):
    def __init__(self, imagepath, region, save_prefix=''):
        self.VGG_MEAN = tf.constant([123.68, 116.78, 103.94], dtype=tf.float32,
                shape=[1,1,3], name="VGG_MEAN")
        """
        1. create an NTMTracker
        2. initialize with the image and region
        """
        self.save_prefix = save_prefix
        self.frame = 0
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
        height, width, _ = image.shape
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
        ratio = FLAGS.bbox_grid/float(FLAGS.cropbox_grid)
        y1 = .5-ratio/2
        x1 = .5-ratio/2
        y2 = .5+ratio/2
        x2 = .5+ratio/2
        self._save_imgs(bbox=[y1,x1,y2,x2])

    def track(self, imagepath):
        self.frame += 1
        #open the image
        image = scipy.misc.imread(imagepath)
        features = self._preprocess_image(self.sess, image, False)
        outputs = self._run_tracker(self.sess, features)
        #print('np', np)
        collected = self._collect_outputs(outputs)
        #print('collected', collected)
        outputs = softmax(collected)
        #TODO: convert this outputs to bbox
        outputs = np.reshape(outputs,
                [FLAGS.cropbox_grid, FLAGS.cropbox_grid])
        self.output_heatmap = outputs
        new_bbox = self._get_bbox(outputs)
        self.output_bbox = new_bbox
        print(new_bbox)
        #import pdb; pdb.set_trace()
        self._save_imgs()
        new_region = self._decode_bbox(new_bbox)
        # use this new bbox to update cropbox
        print(new_region)
        self._update_bbox(self.image_size, new_region)
        return new_region

    def _save_imgs(self, bbox=None):
        if FLAGS.save_img:
            #save and visualize bbox
            if not bbox:
                bbox = self.output_bbox
            img = Image.fromarray(self.cropped_input_image)
            #print('img.size',img.size)
            d = ImageDraw.Draw(img)
            y1,x1,y2,x2 = bbox
            drawbox = [x1*224,y1*224,x2*224,y2*224]
            #print('drawbox',drawbox)
            d.rectangle(drawbox, outline="red")
            img.save('{}outputimg_{}.png'.format(self.save_prefix, self.frame))

            #plot input, output_heatmap together with the visualized bbox
            if hasattr(self, 'output_heatmap'):
                fig, axs = plt.subplots(3,1,figsize=(9,3), dpi=100)
                ax = axs[0]
                ax.imshow(self.cropped_input_image)
                ax.axis('off')

                ax = axs[1]
                ax.imshow(self.output_heatmap)
                ax.axis('off')

                ax = axs[2]
                ax.imshow(img)
                ax.axis('off')
                fig.savefig('{}image_save_{}.png'.format(self.save_prefix,self.frame), bbox_inches='tight', pad_inches=0)
                plt.close(fig)

    def _initial_normal_bbox(self):
        cx = cy = .5
        width = FLAGS.bbox_grid / float(FLAGS.cropbox_grid)
        x1 = cx - width / 2
        x2 = cx + width / 2
        y1 = cy - width / 2
        y2 = cy + width / 2
        return [y1,x1,y2,x2]

    def _contour_to_bbox(self, contour):
        """
        find the convex hull rectangle
        """
        ys, xs = np.transpose(contour)
        y1 = np.min(ys)
        y2 = np.max(ys)
        x1 = np.min(xs)
        x2 = np.max(xs)
        return [y1,x1,y2,x2]

    def _get_bbox(self, outputs, thresholds=np.arange(.01,.1,.01)):
        boxes = []
        initial_bbox = self._initial_normal_bbox()
        for idx, threshold in enumerate(thresholds):
            contours = measure.find_contours(outputs, threshold)

            for n, contour in enumerate(contours):
                normalbbox = preprocess.normalize_bbox(
                        (FLAGS.cropbox_grid, FLAGS.cropbox_grid),
                        self._contour_to_bbox(contour))
                boxes.append((
                    bb_iou(normalbbox, initial_bbox),
                    normalbbox))
        boxes = sorted(boxes)

        #plot the contour boxes for debug
        #fig, axs = plt.subplots(len(boxes),1,figsize=(len(boxes),1), dpi=512)
        #for idx, pair in enumerate(boxes):
        #    ax = axs[idx]
        #    iou, box = pair
        #    y1,x1,y2,x2 = box
        #    scale = FLAGS.cropbox_grid-1
        #    w = (x2-x1)*scale
        #    h = (y2-y1)*scale
        #    x = x1*scale
        #    y = y1*scale
        #    ax.imshow(outputs)
        #    ax.add_patch(
        #            pts.Rectangle(
        #                (x,y),w,h,fill=False))
        #    ax.axis('off')
        #fig.savefig('contours_{}.png'.format(self.frame), bbox_inches='tight', pad_inches=0)
        #plt.close(fig)

        #NMS for boxes
        selected_bbox = boxes[-1][-1]
        #import pdb; pdb.set_trace()
        y1,x1,y2,x2 = selected_bbox
        cx, cy = (x1+x2)*.5, (y1+y2)*.5
        print(cx,cy)
        #import pdb;pdb.set_trace()
        width = FLAGS.bbox_grid / float(FLAGS.cropbox_grid)
        x1 = cx - width / 2
        x2 = cx + width / 2
        y1 = cy - width / 2
        y2 = cy + width / 2
        return [y1,x1,y2,x2]

    def _old_get_bbox(self, outputs):
        """
        outputs in 7x7 format
        """
        x_filter = np.array([[-1,1]])
        y_filter = np.array([[-1],[1]])
        x_derivative = scipy.signal.convolve2d(outputs, x_filter, 'valid')
        y_derivative = scipy.signal.convolve2d(outputs, y_filter, 'valid')
        #first take care of x_derivative
        realxs = []
        for y in xrange(FLAGS.cropbox_grid):
            flag = False
            for x in xrange(FLAGS.cropbox_grid-2):
                #find zero-crossing
                left = x_derivative[y][x]
                right = x_derivative[y][x+1]
                if left >= 0 and right < 0:
                    deltax = left / (left-right)
                    flag = True
                    break
            if flag:
                realx = x+1+deltax
                realxs.append(realx)
        #then take care of y_derivative
        realys = []
        for x in xrange(FLAGS.cropbox_grid):
            flag = False
            for y in xrange(FLAGS.cropbox_grid-2):
                #find zero-crossing
                top = y_derivative[y][x]
                bottom = y_derivative[y+1][x]
                if top >= 0 and bottom < 0:
                    deltay = top / (top-bottom)
                    flag = True
                    break
            if flag:
                realy = y+1+deltay
                realys.append(realy)
        realxs = reject_outliers(np.array(realxs))
        realys = reject_outliers(np.array(realys))
        #get the new center
        cx, cy = np.mean(realxs), np.mean(realys)
        #normalized coordinate
        cx, cy = cx / FLAGS.cropbox_grid, cy / FLAGS.cropbox_grid
        #print(cx, cy)
        #construct a bbox around this center
        width = FLAGS.bbox_grid / float(FLAGS.cropbox_grid)
        x1 = cx - width / 2
        x2 = cx + width / 2
        y1 = cy - width / 2
        y2 = cy + width / 2
        return [y1,x1,y2,x2]

    def _decode_bbox(self, normalized_bbox):
        """
        decode the bbox into real bbox in picture coordinate
        Args:
            normalized_bbox: [y1,x1,y2,x2]

        return:
            vot.Rectangle instance
        """
        #apply the inverse of transformation
        y1,x1,y2,x2 = preprocess.apply_transformation(normalized_bbox,
                np.linalg.inv(self.transformation))

        w,h = self.image_size
        y1,x1,y2,x2 = y1*h,x1*w,y2*h,x2*w
        return vot.Rectangle(x1,y1,x2-x1,y2-y1)

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
        normalized = False
        if x1 < 1 and y1 < 1 and w < 1 and h < 1:
            normalized = True
        bbox = (y1,x1,y1+h,x1+w)
        width, height = image_size
        #print(image_size)
        #print(bbox)
        #normalized bbox
        if not normalized:
            self.normalized_bbox = preprocess.normalize_bbox((width, height), bbox)
        else:
            self.normalized_bbox = bbox
        #print(self.normalized_bbox)
        #cropbox
        self.cropbox = preprocess.calculate_cropbox(self.normalized_bbox,
                FLAGS.cropbox_grid, FLAGS.bbox_grid)
        #print(self.cropbox)
        #transformation to map cropbox to [0,0,1,1]
        self.transformation = preprocess.calculate_transformation(self.cropbox)
        #print(self.transformation)
        #print(preprocess.apply_transformation(self.cropbox, self.transformation))
        #import pdb; pdb.set_trace()

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
        img = self.image_ph - self.VGG_MEAN
        batch_img = tf.image.crop_and_resize(
                tf.expand_dims(img, 0),
                tf.expand_dims(self.cropbox_ph, 0),
                [0], crop_size)
        self.image_after_crop = tf.cast(tf.squeeze(batch_img) + self.VGG_MEAN, tf.uint8)
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
        features, self.cropped_input_image = sess.run([self.features, self.image_after_crop], feed_dict = {
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
            pad = np.zeros((gt.shape[0], 2))
            features = np.concatenate([features, pad], 1)
            features = np.concatenate([features, gt], 1)
        else:
            pad = np.zeros((num_features, 3))
            features = np.concatenate([features, pad], 1)
            #[...,0,0,1]
            frame_delimiter = np.concatenate([
                np.zeros((1, num_channels+1)),
                np.ones((1,1)),
                np.zeros((1, 1))], 1)
            #[...,0,1,0]
            feature_delimiters = np.concatenate([
                np.zeros((num_features, num_channels)),
                np.ones((num_features,1)),
                np.zeros((num_features,2))
                ], 1)
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
count = 1
while True:
    print('processing frame {}'.format(count))
    imagefile = handle.frame()
    if not imagefile:
        break
    region = tracker.track(imagefile)
    handle.report(region)
    count+=1
