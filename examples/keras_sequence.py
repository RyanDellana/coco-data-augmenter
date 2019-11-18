from __future__ import print_function

import os
import numpy as np
import cv2

import keras
from keras.layers import Input, Conv2D, Conv2DTranspose, Concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model

from cocoaugmenter.kerasutil import CocoSeq

"""
Creates an instance of CocoSeq and uses it to train
a simple semantic segmenter to detect cars and people.
"""

def build_segmenter_model(filters=32, output_cnls=2):
    inputs = Input(shape=(64,64,3))
    x = Conv2D(filters=filters, kernel_size=(5,5), activation=None, strides=(1,1), padding='same')(inputs)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2D(filters=filters, kernel_size=(5,5), activation=None, strides=(1,1), padding='same')(x)
    t1 = LeakyReLU(alpha=0.2)(x)
    x = Conv2D(filters=filters*2, kernel_size=(5,5), activation=None, strides=(2,2), padding='same')(t1)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2D(filters=filters*2, kernel_size=(5,5), activation=None, strides=(1,1), padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2D(filters=filters*2, kernel_size=(5,5), activation=None, strides=(2,2), padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2D(filters=filters*2, kernel_size=(5,5), activation=None, strides=(1,1), padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2D(filters=filters*2, kernel_size=(5,5), activation=None, strides=(1,1), padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2D(filters=filters*2, kernel_size=(5,5), activation=None, strides=(1,1), padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2D(filters=filters*2, kernel_size=(5,5), activation=None, strides=(1,1), padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2D(filters=filters*2, kernel_size=(5,5), activation=None, strides=(1,1), padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2D(filters=filters*2, kernel_size=(5,5), activation=None, strides=(1,1), padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2DTranspose(filters=filters, kernel_size=(5,5), activation=None, strides=(4,4), padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    tcat = Concatenate(axis=-1)([t1, x])
    output = Conv2D(filters=output_cnls, kernel_size=(5,5), activation='sigmoid', strides=(1,1), padding='same')(tcat)
    model = Model(inputs, output)
    model.summary()
    return model


if __name__ == '__main__':

    data_dir = '/media/ryan/engrams/data/COCO' # Path to COCO data directory TODO

    seq = CocoSeq(batch_size        = 64,
                  batches_per_epoch = 512,
                  data_dir          = data_dir,
                  class_grps        = [['person'],['car','truck','bus']],
                  grp_probs         = [0.5, 0.5],
                  training          = True,
                  target_width      = 64,
                  min_src_width     = 32,
                  cache_mask_imgs   = False)

    model = build_segmenter_model(output_cnls = 2)

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    history = model.fit_generator(generator = seq,
                                  epochs    = 1,
                                  workers   = 0,
                                  use_multiprocessing = False,
                                  max_queue_size = 512)

    seq.training = False

    score = model.evaluate_generator(generator = seq)

    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    # Generate some sample output. TODO





