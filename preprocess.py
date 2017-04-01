import os
#disable GPU
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw
from multiprocessing import Pool
import scipy.misc
import xml.etree.ElementTree as ET
from tensorflow.python.platform import flags
FLAGS = flags.FLAGS

def join_lists(list_of_lists):
    return [x for sublist in list_of_lists for x in sublist]

class TFCropper(object):
    def __init__(self, crop_size=[224,224]):
        """build the graph"""
        with tf.device('/cpu:0'):
            self.image_ph = tf.placeholder(tf.float32)
            self.bbox_ph = tf.placeholder(tf.float32, shape=[4])
            self.output = tf.squeeze(tf.image.crop_and_resize(
                    tf.expand_dims(self.image_ph, 0),
                    tf.expand_dims(self.bbox_ph, 0),
                    [0], crop_size))

    def __call__(self, image, bbox):
        with tf.Session() as sess:
            output = sess.run(self.output,
                    feed_dict={
                        self.image_ph: image,
                        self.bbox_ph: bbox
                        })
        return output

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def parse_frame(xmlfile):
    xmlroot = ET.parse(xmlfile).getroot()
    """parse size"""
    sizenode = xmlroot.find('size')
    size = [int(sizenode.find('width').text),
            int(sizenode.find('height').text)]
    """parse objects"""
    objnodes = xmlroot.findall('object')
    objs = {}
    for objnode in objnodes:
        trackid = int(objnode.find('trackid').text)
        bboxnode = objnode.find('bndbox')
        #bbox in [y1, x1, y2, x2] format
        bbox = [int(bboxnode.find('ymin').text),
                int(bboxnode.find('xmin').text),
                int(bboxnode.find('ymax').text),
                int(bboxnode.find('xmax').text)]
        objs[trackid] = bbox
    """filename"""
    filename = xmlroot.find('filename').text
    """sequence name"""
    seqname = os.path.basename(xmlroot.find('folder').text)
    return {
            'size': size,
            'objs': objs,
            'filename': filename,
            'seqname': seqname,
           }

def normalize_bbox(size, bbox):
    width, height = size
    y1, x1, y2, x2 = bbox
    return [y1/float(height-1),
            x1/float(width-1),
            y2/float(height-1),
            x2/float(width-1)]

def calculate_cropbox(normalbbox, cropbox_grid, bbox_grid):
    """
    Args:
        bbox: [y1, x1, y2, x2], all are normalized float values
    """
    y1, x1, y2, x2 = normalbbox
    ratio = cropbox_grid/float(bbox_grid)
    """width"""
    x_center = (x1+x2)/2
    width = x2-x1
    cropwidth = ratio * width
    x2n = x_center + cropwidth/2
    x1n = x_center - cropwidth/2
    """height"""
    y_center = (y1+y2)/2
    height = y2-y1
    cropheight = ratio * height
    y2n = y_center + cropheight/2
    y1n = y_center - cropheight/2
    return [y1n, x1n, y2n, x2n]


def calculate_transformation(cropbox):
    """
    calculate transformation.
    input is normalized cropbox
    The output transformation will scale cropbox to [0,0,1,1]
    """
    y1, x1, y2, x2 = cropbox
    width = x2-x1
    height = y2-y1
    transformation = np.array([
        [1/width, 0, -x1/width],
        [0, 1/height, -y1/height],
        [0, 0, 1]])
    return transformation

def apply_transformation(normalbbox, transformation):
    """
    convert a normalized bbox in image space to normalized bbox in
    cropbox space
    """
    y1, x1, y2, x2 = normalbbox
    p1 = np.array([x1, y1, 1])
    p2 = np.array([x2, y2, 1])
    p1n = np.dot(transformation, np.reshape(p1, [3,1]))
    p2n = np.dot(transformation, np.reshape(p2, [3,1]))
    p1n = np.squeeze(p1n)
    p2n = np.squeeze(p2n)
    transformed_bbox = [p1n[1], p1n[0], p2n[1], p2n[0]]
    return transformed_bbox


def calculate_transformation_test():
    cropbox = [.3, .4, .5, .6]
    transformation = calculate_transformation(cropbox)
    transformed_bbox = apply_transformation(cropbox, transformation)
    np.testing.assert_almost_equal(transformed_bbox, [0,0,1,1])
    print("SUCCESS:", transformed_bbox)

def bbox_legal(normalbbox, cropbox, cropbox_grid,
        bbox_grid,
        deform_threshold,
        zoom_threshold):
    """
    make sure normalbbox is within cropbox, and is not too different from
    initial scale and aspect ratio
    """
    within_bound = normalbbox[0] >= cropbox[0] and\
           normalbbox[1] >= cropbox[1] and\
           normalbbox[2] <= cropbox[2] and\
           normalbbox[3] <= cropbox[3]

    y1,x1,y2,x2 = normalbbox
    w, h = x2-x1, y2-y1
    y1,x1,y2,x2 = cropbox
    cw, ch = x2-x1, y2-y1

    whr, hwr = w/h/(cw/ch), h/w/(ch/cw)
    deformed = hwr > 1+deform_threshold or whr > 1+deform_threshold


    y1,x1,y2,x2 = cropbox
    cw, ch = x2-x1, y2-y1
    ratio = bbox_grid/float(cropbox_grid)
    ub, lb = ratio*(1+zoom_threshold), ratio*(1-zoom_threshold)
    zoomed = w/cw > ub or w/cw < lb or h/ch > ub or h/ch < lb

    #print("inbound {}, deformed {}, zoomed {}".format(within_bound, deformed, zoomed))
    return within_bound and (not deformed) and (not zoomed)


def matlab_style_gauss2D(shape=(3,3),sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def discrete_gauss(center=(.5,.5), shape=(7,7), sigma=0.75):
    """
    generate a discrete gaussian centered at 'center' (normalized coordinate)
    of 'shape'-sized grid, with sigma as stdev
    steps:
        1. find out the coordinates of the grid on the coordinate system
        centered at 'center'
    """
    cx, cy = [a*b for a,b in zip (center,shape)]
    w, h = shape
    y, x = np.ogrid[-cy+.5:h-cy+.5, -cx+.5:w-cx+.5]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def discrete_gauss_test():
    np.testing.assert_almost_equal(discrete_gauss(),
            matlab_style_gauss2D((7,7),0.75))
    print("SUCCESS: discrete_gauss_test")


def generate_gt(normalbbox,
        cropbox_grid, bbox_grid):
    """
    given the transformed normalbbox and grid size, generate the ground
    truth
    """
    y1, x1, y2, x2 = normalbbox
    cx = (x1 + x2) / 2.
    cy = (y1 + y2) / 2.
    sigma = bbox_grid/3. #previously it was 4.
    gt = discrete_gauss((cx,cy), (cropbox_grid, cropbox_grid), sigma)
    return gt

def get_img_path_from_anno_path(anno_full_path,
        anno_dir, image_dir):
    anno_relative_path = anno_full_path[len(anno_dir)+1:]
    image_relative_path = anno_relative_path[:-3]+'JPEG'
    image_full_path = os.path.join(image_dir, image_relative_path)
    return image_full_path

def process_sequence(root):
    framefiles = sorted([x for x in os.listdir(root) if x.endswith('.xml')])
    cropboxes = {}
    transformations = {}
    records = {}
    cropper = TFCropper()

    for idx, framefile in enumerate(framefiles):
        #print('processing {}'.format(framefile))
        anno_full_path = os.path.join(root, framefile)
        parsed_frame = parse_frame(anno_full_path)
        size = parsed_frame['size']
        for trackid, bbox in parsed_frame['objs'].items():
            """normalize this boundingbox"""
            normalbbox = normalize_bbox(size, bbox)
            gt = None
            if trackid not in cropboxes:
                """this is the object's first appearance"""
                """calculate cropbox"""
                cropboxes[trackid] = calculate_cropbox(normalbbox,
                        FLAGS.cropbox_grid, FLAGS.bbox_grid)
                """calculate transformation"""
                transformations[trackid] =\
                        calculate_transformation(cropboxes[trackid])
                """record effective frames"""
                records[trackid] = [parsed_frame['filename']]
                """generate gt"""
                transformed_bbox = [
                        .5-FLAGS.bbox_grid/float(FLAGS.cropbox_grid)/2,
                        .5-FLAGS.bbox_grid/float(FLAGS.cropbox_grid)/2,
                        .5+FLAGS.bbox_grid/float(FLAGS.cropbox_grid)/2,
                        .5+FLAGS.bbox_grid/float(FLAGS.cropbox_grid)/2,
                        ]
                gt = generate_gt(transformed_bbox, FLAGS.cropbox_grid,
                        FLAGS.bbox_grid)
            else:
                """
                this object has already appeared in previous frames
                """
                """make sure this normalized bbox is legal"""
                if bbox_legal(normalbbox, cropboxes[trackid],
                        FLAGS.cropbox_grid, FLAGS.bbox_grid,
                        FLAGS.deform_threshold, FLAGS.zoom_threshold):
                    """record effective frame"""
                    records[trackid].append(parsed_frame['filename'])
                    """calculate the gt using previously saved
                    transformation"""
                    transformation = transformations[trackid]
                    transformed_bbox =\
                        apply_transformation(normalbbox, transformation)
                    gt = generate_gt(transformed_bbox, FLAGS.cropbox_grid,
                            FLAGS.bbox_grid)
            if gt is not None:
                """one object one dir"""
                unique_id = parsed_frame['seqname']+'_'+str(trackid)
                output_dir = os.path.join(FLAGS.output_dir, unique_id)
                ensure_dir(output_dir)
                """now save the gt for this object at this frame"""
                assert(gt.dtype == np.float64)
                gt.tofile(os.path.join(output_dir,
                    parsed_frame['filename']+'.bin'))
                """save the metadata for cropping in tensorflow"""
                image_full_path =\
                get_img_path_from_anno_path(anno_full_path,
                        FLAGS.annotation_dir, FLAGS.image_dir)
                with open(os.path.join(output_dir,
                        parsed_frame['filename']+'.txt'), 'w') as f:
                    f.write("{crop[0]},{crop[1]},{crop[2]},{crop[3]},{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]},{image_path}".format(crop=cropboxes[trackid],bbox=transformed_bbox,image_path=image_full_path))

                """save the images for debug use"""
                if FLAGS.save_imgs:
                    gt_resized = scipy.misc.imresize(gt, (224,224), 'nearest')
                    scipy.misc.imsave(os.path.join(output_dir,parsed_frame['filename']+'_gt.png'), gt_resized)
                    """also save the crop of the input image"""
                    img = scipy.misc.imread(image_full_path)
                    cropped = cropper(img, cropboxes[trackid])
                    cropped = Image.fromarray(np.uint8(cropped))
                    d = ImageDraw.Draw(cropped)
                    y1,x1,y2,x2 = transformed_bbox
                    d.rectangle([x1*224,y1*224,x2*224,y2*224], outline="red")
                    cropped.save(os.path.join(output_dir,
                        parsed_frame['filename']+'_crop.png'))
    print('generated {} frames'.format(len(join_lists(records.values()))))



def main():
    """
    for each sequence:
        for each frame:
            if this is zeroth frame:
                for each object in the frame:
                    crop and resize the object to center, and remember the
                    transformation matrices
            else:
                for each transformation matrix:
                    apply the transformation to the corresponding object. If
                    it's out of scope, stop tracing it. by removing the
                    transformation matrix.
            if transformation matrix is empty, break the loop
    """
    if FLAGS.image_dir == "":
        raise Exception("image_dir must be provided")
    if FLAGS.annotation_dir == "":
        raise Exception("annotation_dir must be provided")
    if FLAGS.output_dir == "":
        raise Exception("output_dir must be provided")

    sequences = []
    for root, dirs, files in os.walk(FLAGS.annotation_dir):
        """
        tell if this root correspond to a sequence
        """
        files = [x for x in files if x.endswith('.xml')]
        if len(files) > 0:
            sequences.append(root)
    print('found {} sequences'.format(len(sequences)))
    pool = Pool(7)
    pool.map(process_sequence, sequences, 1000)
    #map(process_sequence, sequences)


if __name__ == '__main__':
    flags.DEFINE_string("image_dir", "", "dir for data")
    flags.DEFINE_string("annotation_dir", "", "dir for data")
    flags.DEFINE_string("output_dir", "", "dir for outputs")
    flags.DEFINE_boolean("save_imgs", False, "flag to indicate whether to save the actual cropped image. If set to true, the bmp formatted crop will be saved. Use this for debugging purpose. [False]")
    flags.DEFINE_boolean("run_test", False, "whether to run tests [False]")
    flags.DEFINE_integer("cropbox_grid", 7, "side length of grid, on which the ground truth will be generated")
    flags.DEFINE_integer("bbox_grid", 3, "side length of bbox grid")
    flags.DEFINE_float("deform_threshold", 0.1, "criterion to stop the producing of bbox")
    flags.DEFINE_float("zoom_threshold", 0.1, "criterion to stop the producing of bbox upon zoom in/out of object")
    if FLAGS.run_test:
        calculate_transformation_test()
        discrete_gauss_test()
    else:
        main()
