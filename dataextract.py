# coding=utf-8
import os
import sys
import random
import tensorflow as tf
import json
from PIL import Image
import csv

#DIRECTORY_IMAGES = '../data/train'
DIRECTORY_IMAGES = '../data/test'
#DIRECTORY_IMAGES = '../data/other'
RANDOM_SEED = 4242
SAMPLES_PER_FILES = 1600


def _process_image(directory, name,writer):
    filename = os.path.join(directory, DIRECTORY_IMAGES, name + '.jpg')
    image_data = tf.gfile.FastGFile(filename, 'rb').read()
    filedir = directory + "/annotations.json"
    ids = open(filedir).read().splitlines()
    annos = json.loads(open(filedir).read())
    annos['imgs'][name]
    for obj in annos['imgs'][name]['objects']:
        label = obj['category']
        bbox = obj['bbox']
        ymin = float(bbox['ymin'])
        xmin = float(bbox['xmin'])
        ymax = float(bbox['ymax'])
        xmax = float(bbox['xmax'])
        writer.writerow((name, xmin, ymin, xmax, ymax, label))

    return


def run(tt100k_root, split):
    split_file_path = os.path.join(tt100k_root, split, 'ids.txt')
    print('>> ', split_file_path)
    i = 0
    #f = open('train.csv', 'w')
    f = open('test.csv', 'w')
    #f = open('other.csv', 'w')
    writer=csv.writer(f)
    with open(split_file_path) as f:
        filenames = f.readlines()
    while i < len(filenames):
        sys.stdout.write('\r>> Converting image %d/%d' % (i + 1, len(filenames)))
        sys.stdout.flush()
        filename = filenames[i].strip()
        _process_image(tt100k_root, filename,writer)
        i += 1
    f.close()
    print('\n>> Finished converting the TT100K %s dataset!' % (split))


if __name__ == '__main__':
    #run('../data', 'train')
    run('../data', 'test')
    #run('../data', 'other')
