from json import load
import numpy as np
from os import path
from random import shuffle
from skimage.measure import block_reduce
import xml.etree.ElementTree as ET
from scipy import misc
from skimage import color
import glob
from shutil import rmtree
import cv2


def data_generator(data_dir, batch_size, input_shape, classes):
    def load_text_annotation(f):
        annotation_file = path.splitext(f)[0] + '_bg.txt'
        with open(annotation_file, 'r') as f:
            d = f.read().splitlines()
        return float(d[-1])

    def load_files(files):
        h, w, ch = input_shape
        X = np.empty((batch_size, h, w, 3))
        Y = np.zeros((batch_size, len(classes) + 1))
        text = []

        for i, f in enumerate(files):
            X[i] = cv2.imread(f[0])
            Y[i] = f[1]

        return X, Y

    def generate_gt(cl, classes, bg_area):
        gt = [0] * (len(classes) + 1)
        gt[classes.index(cl)] = 1 - bg_area
        gt[-1] = bg_area
        return gt

    def load_file_list(base_dir, classes):
        """ Returns dictionary with filenames as keys and ground truth as values"""
        data = []
        for c in classes:
            files = glob.glob(path.join(data_dir, c, "*.png"))
            files = list(filter(lambda x: "_bg" not in x, files))
            bg_area = map(load_text_annotation, files)
            gt = map(lambda x: generate_gt(c, classes, x), bg_area)
            data += list(zip(files, list(gt)))
        return data

    files = load_file_list(data_dir, classes)
    print('Found {} samples'.format(len(files)))
    if len(files) % batch_size != 0:
        print('Warning: len(files) % batch_size != 0')

    if len(files) <= batch_size:
        raise Exception('len(files) <= batch_size')

    while True:
        shuffle(files)
        for i in range(0, len(files) - batch_size + 1, batch_size):
            yield load_files(files[i:i + batch_size])


def main():
    input_shape = (512, 512, 1)
    a = data_generator(
        '../test',
        batch_size=1,
        input_shape=input_shape,
        classes=['good', 'bad'])

    batch = next(a)
    print('batch[0][0].shape =', batch[0][0].shape)
    print('batch[1][0] =', batch[1][0])


if __name__ == "__main__":
    main()
