"""
ILSVRC VID dataset is a multi-obj detection set, but our goal and final testing
benchmark set is a single-obj tracking set.
"""
import os
import pickle
import numpy as np
from ilsvrc_visualizer import get_statistics as get_seq_statistics
from feature_matcher import matches, get_vgg_sizes
from multiprocessing import Pool

VGG_sizes, size_list = get_vgg_sizes()
#layers = [x for x, y in size_list if not x[:2] == 'fc']
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
        get_seq_statistics([
            '/home/jowos/data/ILSVRC2015/Data/VID/train/a',
            '/home/jowos/data/ILSVRC2015/Data/VID/train/b'],
            [
            '/home/jowos/data/ILSVRC2015/Annotations/VID/train/a',
            '/home/jowos/data/ILSVRC2015/Annotations/VID/train/b',
            ])
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
        seq_dir, frame_jpg = os.path.split(frame_path)
        frame_name, _ = os.path.splitext(frame_jpg)
        for obj in objs:
            # a global identifier of the object
            obj_name = seq_dir + '_' + obj['trackid']
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

def gen_sequences(obj_frame_statistics, iou_threshold = 0.5):
    """
    generate viable sequences. Each sequence must satisfy the following
    constraints:
        1. single object. only one object in each sequence
        2. iou_threshold. There must at least be one prior box that has an iou
        with the gt bbox over given threshold. Asserting this make sure we
        only look at invariant scale sequences.

    return:
        an array, of sequences, each of the form:
            (seq_dir, obj_name, subseq_id, seq_length, [frames])
        each frame is of the form:
            (frame_file_name, frame_size, bbox)

    If a raw sequence has certain frames in the middle that don't satisfy the
    constraints, the sequence will be segmented into multiple subsequences.
    Each subsequence will have a subsequence id.
    """
    # sort the statistics by obj_name_frame_name
    # this will result in statistics sorted first in sequence, then in object,
    # then in frame
    obj_frame_statistics.sort(key=lambda x: x[2]+'_'+x[1])
    #print(obj_frame_statistics)
    #raw_input()
    last_obj = None
    last_seq = None
    last_frame = None
    ret = []
    seq = []
    subseq_id = 0
    for seq_dir, frame_name, obj_name, frame_size, bbox, ious in obj_frame_statistics:
        # check if this is a new sequence
        if not (last_obj and last_obj == obj_name):
            #the obj_sequence ends
            #a new obj_sequence starts
            #save whatever we have got
            if len(seq) > 0:
                ret.append((last_seq, last_obj, subseq_id, len(seq), seq))
            #reset the subseq_id and seq buffer
            seq = []
            subseq_id = 0
            #the next line skips the raw sequence break check
            last_frame = None
        # check if there's any break in the raw sequence
        if last_frame:
            num_last_frame = int(last_frame)
            this_frame = int(frame_name)
            assert(this_frame > num_last_frame)
            if this_frame - num_last_frame > 1:
                #there's a skip
                #save whatever we have got
                if len(seq) > 0:
                    ret.append((seq_dir, obj_name, subseq_id, len(seq), seq))
                    seq = []
                    subseq_id += 1
        #count the number of satisfactory overlaps
        count = sum(np.sum(iou > iou_threshold) for iou in ious)
        #add this obj to the sequence if it satisfies
        if count > 0:
            #generate the gt
            gt = [(iou > 0.5) for iou in ious]
            #store this obj-frame into the sequence buffer
            seq.append((os.path.join(seq_dir,frame_name+'.JPEG'),
                frame_size, bbox, gt))
        else: #no, not satisfactory
            if len(seq) > 0: #if there's previous sequence in it
                #save the sequence
                ret.append((seq_dir, obj_name, subseq_id, len(seq), seq))
                seq = []
                #increment the subseq id
                subseq_id += 1
        #remember the seq dir and obj name
        last_seq = seq_dir
        last_obj = obj_name
        #update last_frame information
        last_frame = frame_name
    print("generated {} sequences".format(len(ret)))
    return ret

def main():
    """
    1. calculate the statistics of all object-frames: the matched features and
    their IoU. Save this information as pickle
    2. based on some parameters, select the ideal object-sequence for training
    and validation.
    """
    try:
        with open('obj_seq_statistics.pkl', 'r') as f:
            obj_seq_statistics = pickle.load(f)
    except:
        obj_seq_statistics = objframe_statistics()
        #print obj_seq_statistics
        with open('obj_seq_statistics.pkl', 'w') as f:
            pickle.dump(obj_seq_statistics, f)

    with open('generated_sequences.pkl', 'w') as f:
        pickle.dump(gen_sequences(obj_seq_statistics), f)

if __name__ == '__main__':
    main()
