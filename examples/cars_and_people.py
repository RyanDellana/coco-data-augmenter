from __future__ import print_function

import os
import numpy as np
import cv2

from cocoaugmenter.datagen import CocoDataGen

"""
This example samples evenly from two class groups, automobiles and persons.
Sample images are written to a directory "./augmented_samples/"
"""

dataDir     = '/media/ryan/engrams/data/COCO' # Path to COCO data directory TODO
classGroups = [['person'], 
               ['car', 'truck', 'bus']] # Define class groups
groupProbabilities = [0.5, 0.5] # Define sampling probabilities for class groups
num_samples = 20 # Number of random samples to take for this example

dataGen = CocoDataGen(dataDir   = dataDir,
                      classGrps = classGroups,
                      grpProbs  = groupProbabilities)

# Create directory to store generated images
if not os.path.exists('augmented_samples'):
    os.makedirs('augmented_samples')

# Generate random images according to the defined class group distribution
for idx in range(num_samples):
    print('sample', idx)
    imgTensor, segTensor, metadata = dataGen.sample()
    imgcv2 = (imgTensor*255.0).astype(np.uint8)
    cv2.imwrite('augmented_samples/sample_img'+str(idx)+'.jpg', imgcv2)
    for cnl_idx in range(segTensor.shape[2]):
        imgcv2 = segTensor[:,:,cnl_idx].reshape(segTensor.shape[0:2]).astype(np.uint8)*255
        cv2.imwrite('augmented_samples/sample_img'+str(idx)+'_'+str(cnl_idx)+'.jpg', imgcv2)
