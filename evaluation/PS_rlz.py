#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 13:38:48 2022

@author: Rodrigo
"""

import numpy as np
import matplotlib.pyplot as plt
import pydicom
import pathlib
import argparse
import sys
import pyeval

#%%

#Read Dicom function
def readDicom(dir2Read, imgSize):
    
    # List dicom files
    dcmFiles = list(pathlib.Path(dir2Read).glob('*.dcm'))
    
    dcmImg = np.empty([imgSize[0],imgSize[1]])
    
    if not dcmFiles:    
        raise ValueError('No DICOM files found in the specified path.')

    for dcm in dcmFiles:
        
        ind = int(str(dcm).split('/')[-1].split('_')[-1].split('.')[0])
        
        if ind == 7:
            dcmH = pydicom.dcmread(str(dcm))
            
            dcmImg = dcmH.pixel_array[130:-130,50:-50].astype('float32')  
    
    return dcmImg

#%%

if __name__ == '__main__':
    
    ap = argparse.ArgumentParser(description='Restore low-dose mamography')
    ap.add_argument("--rlz", type=int, required=True, 
                    help="Realization number")
    ap.add_argument("--typ", type=str, required=True, 
                    help="Loss type")
    ap.add_argument("--mas_ld", type=int, required=True, 
                    help="mAs for low-dose")

    
    args = vars(ap.parse_args())
    
    rlz = args['rlz']
    typ = args['typ']
    
    DL_type = 'PS'
    
    path2Read = ''
        
    # Noise scale factor
    mAsFullDose = 60
    mAsLowDose = args['mas_ld']
    
    batch_size = 50
    
    red_factor = mAsLowDose / mAsFullDose
    
    # Image parameters
    nRows = 1788    
    nCols = 1564 

    roiSize = 50
    n_rlzs_fullDose = 10
    
    maskBreast = np.load('data/maskBreast.npy')
           
    
    # DL restored doses
    paths = list(pathlib.Path(path2Read + "Restorations/31_" + str(mAsLowDose) ).glob('DBT_DL_' + DL_type + '*')) 
    if paths == []:
        raise ValueError('No DL results found.')

    restDose_DL_rlzs = np.empty(shape=(nRows,nCols,n_rlzs_fullDose))
    for idZ, path in enumerate(paths):
        # DL restored doses
        restDose_DL_rlzs[:,:,idZ]  = readDicom(path,(nRows,nCols))
                
    ps_DL = np.zeros((26))

    for z in range(n_rlzs_fullDose):        
        _, nps1D, f1D = pyeval.powerSpectrum(restDose_DL_rlzs[:,:,z]*maskBreast, roiSize=roiSize, pixelSize=0.14)
        ps_DL += nps1D
    
    ps_DL /= n_rlzs_fullDose
    
    np.save("data/{}-{}rlz-{}mAs".format(typ,rlz,mAsLowDose), ps_DL)
    
