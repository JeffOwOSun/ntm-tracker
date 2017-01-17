"""
another simple script to visualize the bbox of ilsvrc
"""
import sys
import os
from PIL import Image, ImageDraw
import xml.etree.ElementTree as ET

def get_statistics(image_dir, anno_dir):
    #walk the directories
    statistics = []
    for root, dirs, files in os.walk(anno_dir):
        for filename in files:
            if not filename[-4:] == '.xml':
                continue
            anno_full_path = os.path.join(root, filename)
            anno_relative_path = anno_full_path[len(anno_dir)+1:]
            image_relative_path = anno_relative_path[:-3]+'JPEG'
            image_full_path = os.path.join(image_dir, image_relative_path)
            tree = ET.parse(anno_full_path)
            xmlroot = tree.getroot()
            # a frame
            frame = []
            for child in xmlroot:
                if not child.tag == 'object':
                    continue
                # an object
                obj = {}
                for grandchild in child:
                    if grandchild.tag == 'bndbox':
                        obj['bbox'] = {x.tag: x.text for x in grandchild}
                    else:
                        obj[grandchild.tag] = grandchild.text
                frame.append(obj)
            statistics.append((anno_full_path, image_full_path, obj))
    return statistics

def main():
    if len(sys.argv) < 3:
        print('USAGE: {} path_to_images path_to_annotations'.format(sys.argv[0]))
        return

    image_dir, anno_dir = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
    print(get_statistics(image_dir, anno_dir)[0])


if __name__ == '__main__':
    main()

