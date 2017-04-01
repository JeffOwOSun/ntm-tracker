"""
interface for the VOTchallenge
"""
import os
import pickle
from test_tracker import NTMTracker
from vot import Rectangle
import tensorflow as tf
import numpy as np
from preprocess import calculate_transformation, apply_transformation

def get_image(frame_path):
    txt_path = frame_path + '.txt'
    with open(txt_path, 'r') as f:
        line = f.readline()
    cy1,cx1,cy2,cx2,y1,x1,y2,x2,img_filename = line.split(',')
    cy1,cx1,cy2,cx2 = (float(x) for x in (cy1,cx1,cy2,cx2))
    y1,x1,y2,x2 = (float(x) for x in (y1,x1,y2,x2))
    transformation = calculate_transformation([cy1,cx1,cy2,cx2])
    transformation = np.linalg.inv(transformation)
    y1,x1,y2,x2 = apply_transformation([y1,x1,y2,x2], transformation)
    w,h = x2-x1, y2-y1
    normalized_region = Rectangle(x1,y1,w,h)
    return img_filename, normalized_region

with open('validation_seqs_1700.pkl','r') as f:
    seqs = pickle.load(f)
for idx, seq in enumerate(seqs):
    tf.reset_default_graph()
    seq_path, frame_names = seq
    first_frame = os.path.join(seq_path, frame_names[0])
    imagepath, region = get_image(first_frame)
    tracker = NTMTracker(imagepath, region,
            save_prefix='seq_{}_'.format(idx))
    for frame_name in frame_names[1:]:
        imagepath, region = get_image(os.path.join(seq_path,
            frame_name))
        tracker.track(imagepath)


