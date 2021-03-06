"""
a simple script to visualize the bounding boxes of vot
"""
import sys
import os
from PIL import Image, ImageDraw

def main():
    if len(sys.argv) == 1:
        print('please provide path to the sequence')
        return

    data_path = sys.argv[1]
    gt_name = os.path.join(data_path, 'groundtruth.txt')
    images = sorted([x for x in os.listdir(data_path) if x[-4:] == '.jpg'])
    if len(images) == 0:
        print('the given dir does not contain any images')
        return
    try:
        gt_frames = []
        with open(gt_name, 'r') as f:
            for line in f.readlines():
                frame = []
                coords = map(float, line.split(','))
                for i in xrange(4):
                    frame.append((coords[2*i], coords[2*i+1]))
                gt_frames.append(frame)
    except(x):
        raise x

    assert(len(gt_frames) == len(images))
    output_dir = os.path.join(data_path, 'labeled')
    try:
        os.mkdir(output_dir)
    except OSError:
        pass
    for index in xrange(len(images)):
        img_name = images[index]
        bbox = gt_frames[index]
        img = Image.open(os.path.join(data_path, img_name))
        d = ImageDraw.Draw(img)
        d.polygon(bbox)
        img.save(os.path.join(output_dir, img_name))
    #index = 0
    ##the main loop
    #while True:
    #    img_name = images[index]
    #    bbox = gt_frames[index]
    #    img = Image.open(os.path.join(data_path, img_name))
    #    d = ImageDraw.Draw(img)
    #    d.polygon(bbox)
    #    iqt = ImageQt.ImageQt(img)

    #    """
    #    listen for a key input
    #    """
    #    while True:
    #        print('n: next, p: prev, x: exit')
    #        key = getch()
    #        if key == 'n':
    #            index += 1
    #            break
    #        elif key == 'p':
    #            index -= 1
    #            break
    #        elif key == 'x':
    #            return
    #        else:
    #            continue

if __name__ == '__main__':
    main()
