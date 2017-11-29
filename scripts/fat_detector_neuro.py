#!/usr/bin/env python
# -*- coding: utf-8 -*-

from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Conv2D
from tensorflow.python.keras.layers import Activation
from tensorflow.python.keras.layers import Flatten
from tensorflow.python.keras.layers import concatenate
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.optimizers import SGD, Adam
from tensorflow.python.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.python.keras.applications.resnet50 import ResNet50
from PIL import Image
from os import makedirs
from os import path
import datetime
import subprocess
import sys
from loader import data_generator
import numpy as np
import argparse
from shutil import copy
from os import listdir
import cv2
from math import ceil

batch_size = 10
train_samples_cnt = 20000
test_samples_cnt = 2000
epochs = 1000
input_shape = (512, 512, 3)  # (h, w, ch)
classes = ['good', 'bad', 'background']
n_classes = len(classes)
step_h, step_w = input_shape[:2]


def get_git_revision_hash():
    return subprocess.check_output(['git', 'rev-parse',
                                    'HEAD']).strip().decode()


def create_model(input_shape, n_classes, weights_path, freeze_convnet):
    convnet = ResNet50(
        include_top=False, weights='imagenet', input_shape=input_shape)
    x = convnet.output
    x = Flatten()(x)
    x = Dense(n_classes, activation='softmax', name='out')(x)
    model = Model(inputs=convnet.input, outputs=x)

    # load weights
    if type(weights_path) is str:
        model.load_weights(weights_path, by_name=True)
    elif type(weights_path) is list:
        for p in weights_path:
            model.load_weights(p, by_name=True)

    if freeze_convnet:
        print('Freeze convolutional layers')
        for layer in convnet.layers:
            layer.trainable = False

    return model


def samples_from_dir(folder):
    """Возвращает список файлов в каталоге folder"""
    files = []
    for file in listdir(folder):
        if file.endswith('.png'):
            files.append(path.join(folder, file))
    return files


def tile(img_path, w, h, step_w, step_h):
    """Нарезает исходное изображение, считываемое из файла по адресу img_path
    на тайлы размером w x h с шагом step_w и step_h по осям x и y соответственно.
    Если на изображении не помещается целое количество тайлов, справа и снизу к
    изображению добавляются белые поля.
    Возвращает numpy-массив с тайлами."""
    img = cv2.imread(img_path)
    height, width, _ = img.shape
    total_cols = ceil(width / step_w)
    total_rows = ceil(height / step_h)
    tiles_cnt = total_cols * total_rows
    tiles = np.ones((tiles_cnt, h, w, 3), np.uint8) * 255
    n = 0
    for i in range(0, total_cols * step_w, step_w):
        for j in range(0, total_rows * step_h, step_h):
            j1 = min(j + step_h, img.shape[0])
            i1 = min(i + step_w, img.shape[1])
            tiles[n][0:j1 - j, 0:i1 - i, :] = img[j:j1, i:i1, :]
            n += 1
    return tiles


def main():
    parser = argparse.ArgumentParser(description='Unim test task')
    subparsers = parser.add_subparsers(help='Select what to do', dest='action')
    subparsers.required = True

    # create the parser for the "train" command
    train_parser = subparsers.add_parser('train', help='train model')
    train_parser.add_argument(
        '--weights', help='path to weights file', nargs='+', default=None)
    train_parser.add_argument(
        '--freeze_convnet',
        help='Freeze convolutional layers',
        action='store_true')

    # create the parser for the "predict" command
    predict_parser = subparsers.add_parser(
        'predict', help='predict on directory')
    predict_parser.add_argument(
        '--weights', help='path to weights file', nargs='+', default=None)
    predict_parser.add_argument(
        '-i', '--input', help='Input folder name', required=True)
    predict_parser.add_argument(
        '-o', '--output', help='Output folder name', required=True)

    args = parser.parse_args()

    if 'freeze_convnet' not in args:
        args.freeze_convnet = False

    model = create_model(input_shape, n_classes, args.weights,
                         args.freeze_convnet)

    if args.action == 'train':
        run_id = datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S")
        run_id = run_id + '_commit_' + get_git_revision_hash()
        print("=============================================================")
        print(run_id)
        print("=============================================================")

        epoch_path = 'epoch'
        filename_temp = path.join(epoch_path,
                                  '{}'.format(run_id) + '-{epoch:04d}.hdf5')

        if not path.exists(epoch_path):
            makedirs(epoch_path)

        checkpoint = ModelCheckpoint(
            filepath=filename_temp, save_weights_only=True)

        optimizer = Adam(
        )  #SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy'])

        test_generator = data_generator(
            '../data/test/',
            batch_size=batch_size,
            input_shape=input_shape,
            classes=classes[:-1])

        train_generator = data_generator(
            '../data/train/',
            batch_size=batch_size,
            input_shape=input_shape,
            classes=classes[:-1])

        checkpoint = ModelCheckpoint(
            filepath=filename_temp, save_weights_only=True)

        tensorboard = TensorBoard(
            log_dir='./logs_{}'.format(run_id),
            histogram_freq=0,
            write_graph=True,
            write_images=False)

        model.fit_generator(
            train_generator,
            steps_per_epoch=train_samples_cnt // batch_size,
            validation_data=test_generator,
            validation_steps=test_samples_cnt // batch_size,
            epochs=epochs,
            initial_epoch=0,
            callbacks=[checkpoint, tensorboard])

    elif args.action == 'predict':
        def predict_on_file(filename, k):
            """Осуществляет предсказание класса для изображения, находящегося по
            адресу filename"""
            tiles = tile(filename, *input_shape[:2], step_w, step_h)
            predict = model.predict(tiles, batch_size=32, verbose=0)
            mean_predict = np.mean(predict, axis=0)
            if mean_predict[0] > k * mean_predict[1]:
                return 0, mean_predict
            else:
                return 1, mean_predict

        if not path.exists(args.input):
            raise Exception("Directory not found")
        # Create output directories
        out_dirs = list(
            filter(lambda c: path.join(args.output, c), classes[:-1]))
        for d in out_dirs:
            if not path.exists(d):
                makedirs(d)
        input_files = samples_from_dir(args.input)
        for f in input_files:
            cl, stat = predict_on_file(f, 1)
            dest_name = path.join(out_dirs[cl], path.basename(f))
            stat_file = path.splitext(dest_name)[0] + '.txt'
            copy(f, dest_name)
            with open(stat_file, mode="w") as sf:
                for c, v in zip(classes, stat):
                    sf.write('{}: {}\n'.format(c, v))
            print('File {}: {}'.format(f, classes[cl]))


if __name__ == "__main__":
    main()
