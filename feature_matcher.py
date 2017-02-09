import tensorflow as tf
from PIL import Image
import numpy as np
import os
from vgg import vgg_16
"""
match the features with the ground truth
"""
def get_vgg_sizes():
    inputs = tf.placeholder(tf.float32, shape=(50, 224, 224, 3), name="inputs")
    net, end_points = vgg_16(inputs)
    ret_list = [(x.split('/')[-1], y.get_shape().as_list()) for x,y in
            end_points.iteritems()]
    ret_dict = {x:y for x,y in ret_list}
    return ret_dict, ret_list

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
        heat_map = np.zeros((height, width))
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
                area_i = max(xmax - xmin,0)*max(ymax - ymin,0)
                if area_i == 0:
                    iou = 0
                else:
                    # the area of union
                    area_u = (priorbox[1][0] - priorbox[0][0]) * \
                            (priorbox[1][1] - priorbox[0][1]) + \
                            (bbox[1][0] - bbox[0][0]) * \
                            (bbox[1][1] - bbox[0][1]) - area_i
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
    VGG_sizes, _ = get_vgg_sizes()
    #print(VGG_sizes)
    layer_names = ['conv2_1', 'conv3_3', 'conv4_3', 'conv5_3']
    heatmaps, ious = matches(VGG_sizes, layer_names, (1280, 720),
            [(323, 216), (1050, 428)])
    for idx, iou_map in enumerate(ious):
        Image.fromarray(iou_map).convert('RGB').save(os.path.join('matcher_test', layer_names[idx]+'.jpg'))
        Image.fromarray(heatmaps[idx]).convert('RGB').save(os.path.join('matcher_test', layer_names[idx]+'bin.jpg'))
        #print(iou_map)
        #w, h = iou_map.shape
        #for y in xrange(h):
        #    for x in xrange(w):
        #        if iou_map[y][x] > 0.5:
        #            print('iou: {}'.format(iou_map[y][x]))
    #print(np.array(heatmaps))

if __name__ == '__main__':
    main()
