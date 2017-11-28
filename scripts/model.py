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

batch_size = 10
train_samples_cnt = 20000
test_samples_cnt = 2000
epochs = 1000
input_shape = (512, 512, 3)  # (h, w, ch)
classes = ['good', 'bad']
n_classes = len(classes) + 1


def get_git_revision_hash():
    return subprocess.check_output(['git', 'rev-parse',
                                    'HEAD']).strip().decode()

def create_model(input_shape, n_classes, weights, freeze_convnet):
    convnet = ResNet50(
        include_top=False, weights='imagenet', input_shape=input_shape)
    x = convnet.output
    x = Flatten()(x)
    x = Dense(n_classes, activation='softmax', name='out')(x)
    model = Model(inputs=convnet.input, outputs=x)

    if weights != None:
        model.load_weights(weights)

    if freeze_convnet:
        print('Freeze convolutional layers')
        for layer in convnet.layers:
            layer.trainable = False

    return model

def main():
    parser = argparse.ArgumentParser(description='Unim test task')
    parser.add_argument(
        'action', help='what to do', choices=['train'])
    parser.add_argument(
        'weights', help='path to weights file', nargs='?', default=None)
    parser.add_argument(
        '--freeze_convnet',
        help='Freeze convolutional layers',
        action='store_true')

    args = parser.parse_args()

    run_id = datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S")
    run_id = run_id + '_commit_' + get_git_revision_hash()
    print("=================================================================")
    print(run_id)
    print("=================================================================")

    epoch_path = 'epoch'
    filename_temp = path.join(epoch_path,
                              '{}'.format(run_id) + '-{epoch:04d}.hdf5')

    if not path.exists(epoch_path):
        makedirs(epoch_path)

    checkpoint = ModelCheckpoint(
        filepath=filename_temp, save_weights_only=True)

    model = create_model(input_shape, n_classes, args.weights, args.freeze_convnet)

    if args.action == 'train':
        optimizer = Adam()  #SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy'])

        test_generator = data_generator(
            '../data/test/',
            batch_size=batch_size,
            input_shape=input_shape,
            classes=classes)

        train_generator = data_generator(
            '../data/train/',
            batch_size=batch_size,
            input_shape=input_shape,
            classes=classes)

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


if __name__ == "__main__":
    main()
