from __future__ import print_function

import os
import numpy as np
import cv2

from keras.utils import Sequence

from cocoaugmenter.datagen import CocoDataGen


class CocoSeq(Sequence):

    def __init__(self, 
                 batch_size,
                 batches_per_epoch,
                 data_dir,
                 class_grps,
                 grp_probs,
                 training           = True,
                 target_width       = 128, 
                 min_src_width      = 64,
                 height_shift_range = 0.2,
                 width_shift_range  = 0.2,
                 zoom_range         = (0.5, 1.0),
                 horizontal_flip    = True,
                 cache_mask_imgs    = False):
        self.batch_size         = batch_size
        self.batches_per_epoch  = batches_per_epoch
        self.data_dir           = data_dir
        self.class_grps         = class_grps
        self.grp_probs          = grp_probs
        self.training           = training
        self.target_width       = target_width
        self.min_src_width      = min_src_width
        self.height_shift_range = height_shift_range
        self.width_shift_range  = width_shift_range
        self.zoom_range         = zoom_range
        self.horizontal_flip    = horizontal_flip
        self.cache_mask_imgs    = cache_mask_imgs
        self.data_gen = CocoDataGen(dataDir   = self.data_dir,
                                    classGrps = self.class_grps,
                                    grpProbs  = self.grp_probs,
                                    cacheMaskImgs = self.cache_mask_imgs)

    def __len__(self):
        return self.batches_per_epoch

    def __getitem__(self, idx):
        sample_args = {'training':         self.training,
                       'targetWidth':      self.target_width,
                       'minSrcWidth':      self.min_src_width,
                       'heightShiftRange': self.height_shift_range,
                       'widthShiftRange':  self.width_shift_range,
                       'zoomRange':        self.zoom_range,
                       'horizontalFlip':   self.horizontal_flip}
        batch_x, batch_y = [], []
        for idx in range(self.batch_size):
            img_tensor, seg_tensor, metadata = self.data_gen.sample(**sample_args)
            batch_x.append(img_tensor)
            batch_y.append(seg_tensor)
        return np.array(batch_x), np.array(batch_y)


