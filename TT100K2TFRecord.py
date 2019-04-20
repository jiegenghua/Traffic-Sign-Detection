# coding=utf-8
import os
import sys
import random
import tensorflow as tf
import json
from PIL import Image

#DIRECTORY_IMAGES = '../data/train'
DIRECTORY_IMAGES = '../data/test'
#DIRECTORY_IMAGES = '../data/other'
RANDOM_SEED = 4242
SAMPLES_PER_FILES = 1600


def _process_image(directory, name):
    filename = os.path.join(directory, DIRECTORY_IMAGES, name + '.jpg')
    image_data = tf.gfile.FastGFile(filename, 'rb').read()
    filedir = directory + "/annotations.json"
    ids = open(filedir).read().splitlines()
    annos = json.loads(open(filedir).read())
    annos['imgs'][name]
    red_round_labels = ['p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8', 'p9', 'p10', 'p11', 'p12', 'p13', 'p14', 'p15',
                        'p16', 'p17', 'p18',
                        'p19', 'p20', 'p21', 'p22', 'p23', 'p24', 'p25', 'p26', 'p27', 'p28', 'p29', 'pa10', 'pb', 'pc',
                        'pd',
                        'pe', 'ph3.5', 'pl40', 'pm10', 'pn', 'pne', 'pnl', 'pw3']
    with Image.open(filename) as img:
        shape = [img.height,img.width,3]
    #f = open('train.txt', 'w')
    f = open('test.txt', 'a')
    #f = open('other.txt', 'w')
    f.write(name)
    for obj in annos['imgs'][name]['objects']:
        label = obj['category']
        if label in red_round_labels:
            label = 1
        else:
            label= 0
        bbox = obj['bbox']
        ymin = float(bbox['ymin'])
        xmin = float(bbox['xmin'])
        ymax = float(bbox['ymax'])
        xmax = float(bbox['xmax'])
        line=' '+str(xmin)+' '+str(ymin)+' '+str(xmax)+' '+str(ymax)+' '+ str(label)
        print(name, xmin, ymin,line)
        f.write(line)
    f.write('\n')
    return


def run(tt100k_root, split):
    split_file_path = os.path.join(tt100k_root, split, 'ids.txt')
    print('>> ', split_file_path)
    i = 0
    with open(split_file_path) as f:
        filenames = f.readlines()
    while i < len(filenames):
        sys.stdout.write('\r>> Converting image %d/%d' % (i + 1, len(filenames)))
        sys.stdout.flush()
        filename = filenames[i].strip()
        _process_image(tt100k_root, filename)
        i += 1
    print('\n>> Finished converting the TT100K %s dataset!' % (split))


if __name__ == '__main__':
    #run('../data', 'train')
    run('../data', 'test')
    #run('../data', 'other')