from __future__ import absolute_import
from __future__ import print_function

import os, math
import numpy as np
import cv2
from pycocotools.coco import COCO
import pycocotools.mask as maskUtils


# Note: Turns out loading cached masks from disk is slower than generating them on-the-fly.
# So this function is basically useless. TODO
def _cacheMaskImgs(dataDir, 
                   classGrps, 
                   training, 
                   cocoObj = None):
    """ Caches mask images for each class group given in ``classGrps``.
        Generally speeds up CocoDataGen.sample().
        Mask images are stored in folders created in dataDir.

        # Params
            dataDir: Root directory of coco dataset.
            classGrps: List of lists of strings specifying object classes.
                Each list of classes will be grouped together into one mask image.
            training: If ``True`` then uses training set, otherwise uses validation set.
            cocoObj: Provides option to pass a pre-existing coco dataset object.
    """
    if cocoObj:
        coco = cocoObj
    else:
        annPath = '{}/annotations/instances_{}.json'.format(dataDir, 'train2017' if training else 'val2017')
        coco = COCO(annPath)
    # Get image ids by class group
    catIdsByGrp = [coco.getCatIds(catNms=catNms) for catNms in classGrps]
    imgIdsByGrp = []
    for grpIdx, catIds in enumerate(catIdsByGrp):
        imgIds = []
        for catId in catIds:
            imgIds.extend(coco.getImgIds(catIds=[catId]))
        imgIdsByGrp.append(list(set(imgIds)))
    for grp_idx in range(len(classGrps)):
        folder_name = '_'.join(classGrps[grp_idx]) + ('_train' if training else '_val')
        cache_path = os.path.join(dataDir, folder_name)
        if not os.path.exists(cache_path):
            os.makedirs(cache_path)
        for img_id in imgIdsByGrp[grp_idx]:
            img_info = coco.loadImgs([img_id])[0]
            fname_mask = str(img_id).zfill(12)+'.jpg'
            mask_path = os.path.join(cache_path, fname_mask)
            if not os.path.exists(mask_path): 
                ann_ids = coco.getAnnIds(imgIds=img_id, catIds=catIdsByGrp[grp_idx], iscrowd=False) # Not sure about the iscrowd param TODO
                anns = coco.loadAnns(ann_ids)
                # create an empty mask image
                mask_composite = np.zeros(shape=(img_info['height'], img_info['width']), dtype=np.uint8)
                for ann in anns: # render mask of each ann and add it to the composite image
                    ann_mask = maskUtils.decode(coco.annToRLE(ann))  # get the contours and mask of this instance.
                    mask_composite = np.maximum(mask_composite, ann_mask)
                mask_composite *= 255
                cv2.imwrite(mask_path, mask_composite) # save the composite image


class CocoDataGen():

    def __init__(self, 
                 dataDir, 
                 classGrps, 
                 grpProbs, 
                 cacheMaskImgs=False):
        annPathTrain   = '{}/annotations/instances_{}.json'.format(dataDir, 'train2017')
        annPathVal     = '{}/annotations/instances_{}.json'.format(dataDir, 'val2017')
        self.dataDir   = dataDir
        self.cocoTrain = COCO(annPathTrain)
        self.cocoVal   = COCO(annPathVal) # Need to do this for validation too. TODO
        self.cacheMaskImgs = cacheMaskImgs
        if self.cacheMaskImgs:
            _cacheMaskImgs(dataDir, classGrps, True, self.cocoTrain)
            _cacheMaskImgs(dataDir, classGrps, False, self.cocoVal)
        self.classGrps = classGrps
        self.grpProbs  = grpProbs
        self.catIdsByGrp = [self.cocoTrain.getCatIds(catNms=catNms) for catNms in classGrps]
        self.imgIdsByGrp = []
        for grpIdx, catIds in enumerate(self.catIdsByGrp):
            imgIds = []
            for catId in catIds:
                imgIds.extend(self.cocoTrain.getImgIds(catIds=[catId]))
            self.imgIdsByGrp.append(list(set(imgIds)))

    def _resize_pad(self, img, dim, interp=None):
        h, w = img.shape[0:2]
        img_zeros = np.zeros((dim, dim, img.shape[2]) if len(img.shape) == 3 else (dim, dim), np.uint8)
        resize_percent = float(dim) / max(h, w)
        w_, h_ = int(math.ceil(w*resize_percent)), int(math.ceil(h*resize_percent))
        if interp is None:
            img_resized = cv2.resize(img, (w_, h_))
        else:
            img_resized = cv2.resize(img, (w_, h_), interpolation=interp)
        img_zeros[(dim-h_)/2:(dim-h_)/2+h_, (dim-w_)/2:(dim-w_)/2+w_] = img_resized[:,:]
        return img_zeros
    
    def _map(self, x, in_min, in_max, out_min, out_max):
        return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

    def sample(self, 
               training         = True,
               targetWidth      = 128, 
               minSrcWidth      = 32,
               heightShiftRange = 0.2,
               widthShiftRange  = 0.2,
               zoomRange        = (0.5, 1.0),
               horizontalFlip   = True):
        """ Extract random image containing one or more of the classes in ``classIds``.
            # Params
                cocoObj: COCO API object.
                classIDs: list of lists of coco class IDs.
                classProbs: list of probabilities of sampling each class (must add up to 1.0).
                classImgIDs: dict of lists of imageIDs that contain at least one instance of the corresponding class (key)
            # Returns
                Square image tensor. (numpy array of type float32).
                Mask tensor with number of channels equal to the number of classes.
                Metadata dictionary (ex. number of instances of each class, coordinates of faces)
        """
        cocoObj = self.cocoTrain if training else self.cocoVal

        def chooseRefAnn():
            # Pick a reference class group according the the probabilities.
            classGrpIdx = np.random.choice(a=len(self.catIdsByGrp), p=self.grpProbs)
            # pick an imageID from the set of images containing class instances of the selected class group.
            classGrpImgIds = self.imgIdsByGrp[classGrpIdx]
            imgId = np.random.choice(classGrpImgIds)
            # get annotation IDs of all instances of classes in chosen group found in the image.
            annIdsByGrp = []
            for classIds_ in self.catIdsByGrp:
                annIdsByGrp.append(cocoObj.getAnnIds(imgIds=imgId, catIds=classIds_, iscrowd=False)) # Not sure about the iscrowd param TODO
            annIds = annIdsByGrp[classGrpIdx]
            # randomly select one of the instances as a reference.
            refAnnId = np.random.choice(annIds)
            refAnn = cocoObj.loadAnns([refAnnId])[0]
            # get metadata about the chosen instance to verify that it's within tolerance.
            rle = cocoObj.annToRLE(refAnn)  # get the contours of this instance.
            (x, y, w, h) = maskUtils.toBbox(rle)
            if int(max(w, h)) < minSrcWidth: # if ROI width is below the minimum.
                return None # reject.
            else:
                return imgId, annIdsByGrp, refAnn, rle, (x, y, w, h)

        ref = chooseRefAnn()
        while ref is None:
            ref = chooseRefAnn()
        imgId, annIdsByGrp, refAnn, rle, (x, y, w, h) = ref
        imgInfo = cocoObj.loadImgs([imgId])[0]
        #area = maskUtils.area(rle)
        (x, y, w, h) = (int(x), int(y), int(w), int(h))
        width = int(max(w, h))
        cx, cy = (int(x + w/2.0), int(y + h/2.0)) # calculate center of instance. (Note: Might now fall on mask region).
        # calculate a square bounding box based on the rectangular bounding box.
        sx, sy = cx-int(width/2.0), cy-int(width/2.0) # square bounding box x and y, width is also the height.
        sx, sy = max(sx, 0), max(sy, 0) # make sure the upper left corner of the box hasn't gone off the image.
        # perform shift augmentation.
        xShiftPixels = int(width * (widthShiftRange - (np.random.random() * 2.0*widthShiftRange)))
        yShiftPixels = int(width * (heightShiftRange - (np.random.random() * 2.0*heightShiftRange)))
        cx += xShiftPixels
        cy += xShiftPixels
        cx, cy = min(max(cx, 0), imgInfo['width']), min(max(cy, 0), imgInfo['height'])
        # perform zoom augmentation.
        zoomMin, zoomMax = zoomRange # i.e. -zoomOutMax, zoomInMax
        zoomMin = max(zoomMin, 0.01)
        zoom = (-1.0/zoomMin) + (np.random.random() * (1.0/zoomMin + zoomMax))
        if zoom > 0.0:
            zoom = self._map(zoom, 0.0, zoomMax, 1.0, zoomMax)
        else:
            zoom = self._map(zoom, -1.0/zoomMin, 0.0, zoomMin, 1.0) 
        # divide width by zoom value and then recalculate sx and sy.
        if int(width/zoom) >= minSrcWidth: # if zooming didn't shrink the roi square too much:
            width = int(width/zoom)
            sx, sy = cx-int(width/2.0), cy-int(width/2.0) # square bounding box x and y, width is also the height.
            sx, sy = max(sx, 0), max(sy, 0) # make sure the upper left corner of the box hasn't gone off the image.
        if not self.cacheMaskImgs:
            # Find set of instances which have bounding rectangles that overlap with bounding box of reference instance:
            annsByGrp = [cocoObj.loadAnns(grpAnnIds) for grpAnnIds in annIdsByGrp]
            overlappingAnnsByGrp = []
            for grpIdx, grpAnns in enumerate(annsByGrp):
                overlappingAnns = []
                for ann in grpAnns:
                    rle_ = cocoObj.annToRLE(ann)  # get the contours of this instance.
                    (x_, y_, w_, h_) = maskUtils.toBbox(rle_)
                    x_overlap = (x_ >= sx and x_ <= sx + width) or (sx >= x_ and sx <= x_ + w_)
                    y_overlap = (y_ >= sy and y_ <= sy + width) or (sy >= y_ and sy <= y_ + h_)
                    if x_overlap and y_overlap: # if overlap:
                        overlappingAnns.append(ann)
                        # Note: It's okay if it's the refAnn, since that overlaps with itself.
                overlappingAnnsByGrp.append(overlappingAnns)
            # Render masks for all instances in the aft-mentioned set:
            maskTensor = np.zeros(shape=(imgInfo['height'], imgInfo['width'], len(self.catIdsByGrp)), dtype=np.uint8)
            for grpIdx, grpAnns in enumerate(overlappingAnnsByGrp):
                # render mask and binary_or add it to the correct maskTensor channel.
                for ann in grpAnns:
                    #print('adding instance to group', grpIdx)
                    annMask = maskUtils.decode(cocoObj.annToRLE(ann))  # get the contours and mask of this instance.
                    #maskTensor[:,:,grpIdx] = np.bitwise_or(maskTensor[:,:,grpIdx], annMask)
                    maskTensor[:,:,grpIdx] = np.maximum(maskTensor[:,:,grpIdx], annMask) # add instance mask to appropriate channel.
                    # ^^^ Consider memoization to speed this up. TODO
        else: # load mask images from files instead
            maskTensor = np.zeros(shape=(imgInfo['height'], imgInfo['width'], len(self.catIdsByGrp)), dtype=np.uint8)
            for grpIdx, classNames in enumerate(self.classGrps):
                folder_name = '_'.join(classNames) + ('_train' if training else '_val')
                cache_path = os.path.join(self.dataDir, folder_name)
                fname_mask = str(imgId).zfill(12)+'.jpg'
                mask_path = os.path.join(cache_path, fname_mask)
                if os.path.exists(mask_path):
                    try:
                        maskTensor[:,:,grpIdx] = cv2.imread(mask_path, False)[:,:]
                    except:
                        print('Error reading mask image:', mask_path)
            maskTensor = np.clip(maskTensor/100, 0, 1)
        # Load the image:
        fname   = imgInfo['file_name']
        imgsDir = '{}/images/{}'.format(self.dataDir, 'train2017' if training else 'val2017')
        imgPath = os.path.join(imgsDir, fname)
        imgcv2  = cv2.imread(imgPath)
        # Crop out the region bounded by the square:
        imgRoi  = imgcv2[sy:sy+width, sx:sx+width]
        # Crop out the corresponding region in the mask tensor:
        RoiMaskTensor = maskTensor[sy:sy+width, sx:sx+width, :]
        # Resize the roi and masks:
        if not (targetWidth is None):
            if sy+width > imgInfo['height'] or sx+width > imgInfo['width']:
                imgRoi = self._resize_pad(imgRoi, targetWidth)
                RoiMaskTensor = self._resize_pad(RoiMaskTensor, targetWidth)
            else:
                imgRoi = cv2.resize(imgRoi, (targetWidth, targetWidth))
                RoiMaskTensor = cv2.resize(RoiMaskTensor, (targetWidth, targetWidth))
        # Randomly mirror if enabled:
        if horizontalFlip and np.random.random() > 0.5:
            #print('flip.')
            imgRoi = cv2.flip(imgRoi, 1)
            RoiMaskTensor = cv2.flip(RoiMaskTensor, 1)
        RoiTensor = imgRoi.astype(np.float32) / 255.0
        RoiMaskTensor = RoiMaskTensor.astype(np.float32)
        # Decided to exclude the meta-data info since it's not consistent or accurate anyway.
        return RoiTensor, RoiMaskTensor, None
                                                                                                                                     
