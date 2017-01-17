import tensorflow as tf
import numpy as np
from vgg import vgg_16
"""
match the features with the ground truth
"""
def get_vgg_sizes():
    inputs = tf.placeholder(tf.float32, shape=(50, 224, 224, 3), name="inputs")
    net, end_points = vgg_16(inputs)
    return {x.split('/')[-1]:y.get_shape().as_list() for x,y in end_points.iteritems()}

def matches(layer_dims, layers, img_size, bbox, threshold = 0.5):
    """
    args:
        layer_dims: a dictionary from layer name to dimension of layer output tensors.

        layers: an array of layer names to match with ground truth

        img_size: a tuple (width, height) of input image

        bbox: a list of 2 tuples [(x0,y0),(x1, y1)] of the top-left and
        bottom-right vertices of bbox

        threshold: the iou threshold for positive example

    return:
        a stack of index matrices, marking the corresponding features as
        positive and negative examples. The order of which follows that of layers.
    """
    heat_maps = []
    ious = []
    for layer_name in layers:
        im_w, im_h = img_size
        _, height, width, _ = layer_dims[layer_name]
        heat_map = - np.ones((height, width))
        iou_map = np.ndarray(shape=(height, width), dtype=float)
        for x in xrange(width):
            for y in xrange(height):
                priorbox = [
                    (x*im_w/float(width), y*im_h/float(height)),
                    ((x+1)*im_w/float(width), (y+1)*im_h/float(height))]
                # get the intersection rectangle
                xmin = max(priorbox[0][0], bbox[0][0])
                xmax = min(priorbox[1][0], bbox[1][0])
                ymin = max(priorbox[0][1], bbox[0][1])
                ymax = min(priorbox[1][1], bbox[1][1])
                # calculate the area of intersection rectangle
                area_i = max((xmax - xmin)*(ymax - ymin), 0)
                # the area of union
                area_u = (priorbox[1][0] - priorbox[0][0]) * \
                        (priorbox[1][1] - priorbox[0][1]) + \
                        (bbox[1][0] - bbox[0][0]) * \
                        (bbox[1][1] - bbox[0][1])
                #IoU
                iou = area_i / float(area_u)
                #print('iou: {}'.format(iou))
                iou_map[y][x] = iou
                if iou > 0.5:
                    # this is a positive example
                    heat_map[y][x] = 1
        heat_maps.append(heat_map)
        ious.append(iou_map)
    return heat_maps, ious


def main():
    VGG_sizes = get_vgg_sizes()
    #print(VGG_sizes)
    heatmaps, ious = matches(VGG_sizes, ['conv2_1', 'conv3_3', 'conv4_3', 'conv5_3'], (1280, 720),
            [(323, 216), (1050, 428)])
    for iou_map in ious:
        w, h = iou_map.shape
        for x in xrange(w):
            for y in xrange(h):
                if iou_map[y][x] > 0.5:
                    print('iou: {}'.format(iou_map[y][x]))
    #print(heatmaps)

if __name__ == '__main__':
    main()
