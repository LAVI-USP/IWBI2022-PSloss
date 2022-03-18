#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 13:38:48 2022

@author: Rodrigo
"""

import numpy as np
import pydicom as dicom
import h5py
import random
import matplotlib.pyplot as plt

from pathlib import Path
from scipy.ndimage import median_filter, uniform_filter1d
from skimage.filters import threshold_otsu

#%%

def get_img_bounds(img):
    '''Get image bounds of the segmented breast from the FD'''
    
    # for normal dose
    vertProj = np.sum(img,axis=0)
    
    # Smooth the signal and take the first derivative
    firstDv = np.gradient(uniform_filter1d(median_filter(vertProj,size=250), size=50))
    
    # Smooth the signal and take the second derivative
    secondDv = np.gradient(uniform_filter1d(firstDv, size=50))
    
    # Takes its max second derivative
    indX = np.argmin(secondDv) 
    
    # w
    w_min, w_max = 50, img.shape[0]-50
    
    # h
    h_min, h_max = indX, vertProj.shape[0]
    
    thresh = threshold_otsu(img[w_min:w_max, h_min:h_max])
        
    return w_min, h_min, w_max, h_max, thresh


def extract_rois(img_ld, img_fd):
    '''Extract low-dose and full-dose rois'''
    
    # Check if images are the same size
    assert img_ld.shape == img_fd.shape, "image sizes differ"
    
    global trow_away
    
    # Get image bounds of the segmented breast from the GT
    w_min, h_min, w_max, h_max, thresh = get_img_bounds(img_fd)
    
    # Crop all images
    img_ld = img_ld[w_min:w_max, h_min:h_max]
    img_fd = img_fd[w_min:w_max, h_min:h_max]
    
    # Get updated image shape
    w, h = img_ld.shape
    
    rois = []
    
    # Non-overlaping roi extraction
    for i in range(0, w-64, 64):
        for j in range(0, h-64, 64):
            
            # Extract roi
            roi_tuple = (img_ld[i:i+64, j:j+64], img_fd[i:i+64, j:j+64])
            
            # Am I geting at least one pixel from the breast?
            if np.sum(roi_tuple[1] > thresh) == 0:
                if np.sum(roi_tuple[1] > 2000) > 0:
                    trow_away += 1
                else:
                    rois.append(roi_tuple)
            else:
                trow_away += 1                

    return rois


def process_each_folder(folder_name, num_proj=15):
    '''Process DBT folder to extract low-dose and full-dose rois'''
        
    rois = []
    
    # Loop on each projection
    for proj in range(num_proj):
        
        # Low-dose image
        ld_file_name = folder_name + "/{}_L{}.dcm".format(proj,redFactor)

        # Full-dose image
        fd_file_name = folder_name + "/{}.dcm".format(proj)
    
        img_ld = dicom.read_file(ld_file_name).pixel_array
        img_fd = dicom.read_file(fd_file_name).pixel_array
    
        rois += extract_rois(img_ld, img_fd)
                    
    return rois

def check_min_max(imgs):
    '''Check min and max from rois'''
    
    global min_global_img, max_global_img
    
    min_img = np.min(rois)
    max_img = np.max(rois)
    
    if min_img < min_global_img:
        min_global_img = min_img
    if max_img > max_global_img:
        max_global_img = max_img
        
    return

#%%

if __name__ == '__main__':
    
    path2read = ''
    path2write = '../data/'
    
    folder_names = [str(item) for item in Path(path2read).glob("*/*") if Path(item).is_dir()]
    
    folder_names = []
    
    redFactor = 50
    
    mAsFullDose = 60
    mAsLowDose = int(mAsFullDose * (redFactor/100))
    
    nROIs_total = 256000
    
    np.random.seed(0)
    
    trow_away = 0
    flag_final = 0
    nROIs = 0
    
    min_global_img = np.inf
    max_global_img = 0
    
    # Create h5 file
    f = h5py.File('{}DBT_training_{}mAs.h5'.format(path2write, mAsLowDose), 'a')
    
    # Loop on each DBT folder (projections)
    for idX, folder_name in enumerate(folder_names):
        
        # Get low-dose and full-dose rois
        rois = process_each_folder(folder_name)        
                
        data = np.stack([x[0] for x in rois])
        target = np.stack([x[1] for x in rois])
        
        data = np.expand_dims(data, axis=1) 
        target = np.expand_dims(target, axis=1) 
        
        nROIs += data.shape[0]
        
        # Did I reach the expected size (nROIs_total)?
        if  nROIs >= nROIs_total:
            flag_final = 1
            diff = nROIs_total - nROIs
            data = data[:diff,:,:,:]
            target = target[:diff,:,:,:]
                            
        if idX == 0:
            f.create_dataset('data', data=data, chunks=True, maxshape=(None,1,64,64))
            f.create_dataset('target', data=target, chunks=True, maxshape=(None,1,64,64)) 
        else:
            f['data'].resize((f['data'].shape[0] + data.shape[0]), axis=0)
            f['data'][-data.shape[0]:] = data
            
            f['target'].resize((f['target'].shape[0] + target.shape[0]), axis=0)
            f['target'][-target.shape[0]:] = target
            
        print("Iter {};'data' shape:{} and 'target':{}".format(idX,f['data'].shape,f['target'].shape))

        check_min_max(data)
        check_min_max(target)

        if flag_final:
            break

    f.close()  
    print("Min:{}, Max:{}".format(min_global_img, max_global_img))     
     
    
    