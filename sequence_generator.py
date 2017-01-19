"""
ILSVRC VID dataset is a multi-obj detection set, but our goal and final testing
benchmark set is a single-obj tracking set.
"""
import os
import pickle
from ilsvrc_visualizer import get_statistics as get_seq_statistics
from feature_matcher import matches, get_vgg_sizes
from multiprocessing import Pool

VGG_sizes, size_list = get_vgg_sizes()
layers = ['conv4_3']

def distrib_match(obj_frame):
    """
    3. match obj_frames with vgg features
    """
    seq_dir, frame_name, obj_name, frame_size, bbox = obj_frame
    _, ious = matches(VGG_sizes, layers, frame_size, bbox)
    return (seq_dir, frame_name, obj_name, frame_size, bbox, ious)

def objframe_statistics():
    """
    1. get all the raw statistics of sequences
    """
    print('getting raw statistics...')
    try:
        with open('raw_statistics.pkl', 'r') as f:
            raw_statistics = pickle.load(f)
    except:
        raw_statistics = \
        get_seq_statistics('/home/jowos/5.5t/ILSVRC2015/Data/VID/train/a',
            '/home/jowos/5.5t/ILSVRC2015/Annotations/VID/train/a')
        with open('raw_statistics.pkl', 'w') as f:
            pickle.dump(raw_statistics, f)

    """
    2. extract all the obj_frame
    """
    print('extracting object frames')
    obj_frames = []
    for idx, frame in enumerate(raw_statistics):
        if idx % 1000 == 0:
            print('processed {} frames, {} remaining'.format(idx,
                len(raw_statistics) - idx))

        xml_path, frame_path, frame_size, objs = frame
        # split the sequence dir and frame name
        seq_dir, frame_xml = os.path.split(xml_path)
        frame_name, _ = os.path.splitext(frame_xml)
        for obj in objs:
            obj_name = frame_name + '_' + obj['trackid']
            bbox = obj['bbox']
            bbox = [(bbox['xmin'],bbox['ymin']), (bbox['xmax'],bbox['ymax'])]
            obj_frames.append((seq_dir, frame_name, obj_name, frame_size, bbox))


    """
    3 matching
    """
    print('matching obj frames with vgg features')
    pool = Pool(7)
    print(len(obj_frames), obj_frames[0])
    records = pool.map(distrib_match, obj_frames, 1000)

    return records


def main():
    """
    1. calculate the statistics of all object-frames: the matched features and
    their IoU. Save this information as pickle
    2. based on some parameters, select the ideal object-sequence for training
    and validation.
    """
    obj_seq_statistics = objframe_statistics()
    #print obj_seq_statistics
    with open('obj_seq_statistics.pkl', 'w') as f:
        pickle.dump(obj_seq_statistics, f)

if __name__ == '__main__':
    main()
