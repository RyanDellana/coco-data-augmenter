from __future__ import print_function

import os, time
import numpy as np
import cv2

from cocoaugmenter.datagen import CocoDataGen

"""
This example samples evenly from two class groups, automobiles and persons.
The resulting data is written to an npz file.
"""

dataDir     = '/media/ryan/engrams/data/COCO' # Path to COCO data directory TODO
classGroups = [['person'], 
               ['car', 'truck', 'bus']] # Define class groups
groupProbabilities = [0.5, 0.5] # Define sampling probabilities for class groups
training    = False
num_samples = 64*50 # batch size of 64 with 1000 batches in an epoch.

np.random.seed(42)

dataGen = CocoDataGen(dataDir   = dataDir,
                      classGrps = classGroups,
                      grpProbs  = groupProbabilities)

sample_args = {'training'         : training,
               'targetWidth'      : 128, 
               'minSrcWidth'      : 64,
               'heightShiftRange' : 0.2,
               'widthShiftRange'  : 0.2,
               'zoomRange'        : (0.5, 1.0),
               'horizontalFlip'   : True}

# Generate random images according to the defined class group distribution
x_train, y_train = [], []
for idx in range(num_samples):
    print('sample', idx)
    imgTensor, segTensor, metadata = dataGen.sample(**sample_args)
    x_train.append(imgTensor)
    y_train.append(segTensor)
path = os.path.join('val_3200_128px.npz')
np.savez_compressed(path, x_train = np.array(x_train), y_train = np.array(y_train))
