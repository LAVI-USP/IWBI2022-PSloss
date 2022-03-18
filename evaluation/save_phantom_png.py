#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 13:38:48 2022

@author: Rodrigo
"""

import pydicom
import matplotlib.pyplot as plt
import numpy as np
import pathlib
import os
import argparse

#%% Read Dicom function
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

            dcmImg[:,:] = dcmH.pixel_array[130:-130,50:-50].astype('float32')  
    
    return dcmImg

def mat2gray(img, rangeVal):
    
    img = np.clip(img, rangeVal[0], rangeVal[1])
    
    img = (img - rangeVal[0]) / ((rangeVal[1] - rangeVal[0]) / 255.) 
        
    return img    

def save_figure(img, fname):

    img = np.uint8(255 - mat2gray(img,(524, 683)))
    plt.imsave('phantom_ROI/' + fname + '.png', img, cmap=plt.cm.gray)
    

if __name__ == '__main__':
    
    ap = argparse.ArgumentParser(description='Restore low-dose mamography')
    ap.add_argument("--rlz", type=int, required=True, 
                    help="Realization number")
    ap.add_argument("--typ", type=str, required=True, 
                    help="Loss type")

    
    args = vars(ap.parse_args())
    
    rlz = args['rlz']
    typ = args['typ']
    
    DL_types = ['PS']

    path2Read = ''
    
    #%% Image parameters
    
    nRows = 1788   
    nCols = 1564 
    
    mAsFullDose = 60
    reducFactors = [50]
    
    n_all_fullDose = 20
    n_rlzs_fullDose = 10
    n_rlzs_GT = 10
    DL_ind = 0
    
    ii_phantom, jj_phantom  = 1397, 1499
    
    w = 50
    
    if not os.path.exists('phantom_ROI'):
        os.mkdir('phantom_ROI')


    #%% Read clinical data
    myInv = lambda x : np.array((1./x[0],-x[1]/x[0]))
    
    nreducFactors = len(reducFactors)
    
    # Read all full dose images
    fullDose_all = np.empty(shape=(nRows,nCols,n_all_fullDose))
    
    paths = sorted(list(pathlib.Path(path2Read + "31_" + str(mAsFullDose) ).glob('*/')))
    
    if paths == []:
        raise ValueError('No FD results found.') 
    
    print('Reading FD images...')
    for idX, path in enumerate(paths):
        fullDose_all[:,:,idX] = readDicom(path,(nRows,nCols))
       
        
    # Generate the GT - fisrt time
    groundTruth = np.mean(fullDose_all[:,:,n_rlzs_GT:], axis=-1)
    # Generate a boolean mask for the breast
    maskBreast = groundTruth < 2500
    
    # Normalize the GT realizations
    for z in range(n_rlzs_GT, n_all_fullDose):
        unique_rlzs = fullDose_all[:,:,z]
        unique_rlzs = np.polyval(myInv(np.polyfit(groundTruth[maskBreast], unique_rlzs[maskBreast], 1)), unique_rlzs)
        fullDose_all[:,:,z] = np.reshape(unique_rlzs, (nRows,nCols))
        
    # Generate again the GT after normalization
    groundTruth = np.mean(fullDose_all[:,:,n_rlzs_GT:], axis=-1)
    
    # Normalize the full dose realizations
    z=0
                  
    
    # Read MB, restored results and reuced doses
    for reduc in reducFactors:
        
        mAsReducFactors = int((reduc / 100) * mAsFullDose)
        
        # # Reduced doses
        # paths = list(pathlib.Path(path2Read + "31_" + str(mAsReducFactors) ).glob('*')) 
        # if paths == []:
        #     raise ValueError('No RD results found.')
        # paths = [path for path in paths if not 'MB' in str(path) and not 'DL' in str(path)]
        # reduDose_rlzs = np.empty(shape=(nRows,nCols))
        # for idX, path in enumerate(paths):
        #     all_rlzs =  readDicom(path,(nRows,nCols))
        #     # Reduced doses
        #     unique_rlzs = all_rlzs
        #     unique_rlzs = np.polyval(myInv(np.polyfit(groundTruth[maskBreast], unique_rlzs[maskBreast], 1)), unique_rlzs)
        #     reduDose_rlzs = np.reshape(unique_rlzs, (nRows,nCols))
        #     # reduDose_rlzs = ((reduDose_rlzs - 50) / (reduc / 100)) + 50
        #     save_figure(reduDose_rlzs[ii_phantom:ii_phantom+w,jj_phantom:jj_phantom+w], '{}_01_Mammo_R_CC'.format(mAsReducFactors))
        #     # print(reduDose_rlzs[ii_phantom:ii_phantom+w,jj_phantom:jj_phantom+w].min(),reduDose_rlzs[ii_phantom:ii_phantom+w,jj_phantom:jj_phantom+w].max())
        #     break
                
    
        # # MB restored doses
        # paths = list(pathlib.Path(path2Read + "Restorations/31_" + str(mAsReducFactors) ).glob('MB*')) 
        # if paths == []:
        #     raise ValueError('No MB results found.')
        # restDose_MB_rlzs = np.empty(shape=(nRows,nCols))
        # for idX, path in enumerate(paths):
        #     all_rlzs =  readDicom(path,(nRows,nCols))
        #     # MB restored doses
        #     unique_rlzs = all_rlzs
        #     unique_rlzs = np.polyval(myInv(np.polyfit(groundTruth[maskBreast], unique_rlzs[maskBreast], 1)), unique_rlzs)
        #     restDose_MB_rlzs = np.reshape(unique_rlzs, (nRows,nCols))
            
        #     save_figure(restDose_MB_rlzs[ii_phantom:ii_phantom+w,jj_phantom:jj_phantom+w], 'MB_{}_01_Mammo_R_CC'.format(mAsReducFactors))
        #     # print(restDose_MB_rlzs[ii_phantom:ii_phantom+w,jj_phantom:jj_phantom+w].min(),restDose_MB_rlzs[ii_phantom:ii_phantom+w,jj_phantom:jj_phantom+w].max())
        #     break
    
    
        # Loop through DL methods
        for indDL, DL_type in enumerate(DL_types):
            
                print('Reading and calculating {}({}mAs) images...'.format(DL_type,mAsReducFactors))
                
                # DL restored doses
                paths = list(pathlib.Path(path2Read + "Restorations/31_" + str(mAsReducFactors) ).glob('DBT_DL_' + DL_type + '*')) 
                if paths == []:
                    raise ValueError('No DL results found.')
                restDose_DL_rlzs = np.empty(shape=(nRows,nCols))
                for idZ, path in enumerate(paths):
                    all_rlzs =  readDicom(path,(nRows,nCols))
                    
                    # DL restored doses
                    unique_rlzs = all_rlzs
                    unique_rlzs = np.polyval(myInv(np.polyfit(groundTruth[maskBreast], unique_rlzs[maskBreast], 1)), unique_rlzs)
                    restDose_DL_rlzs = np.reshape(unique_rlzs, (nRows,nCols))
                    
                    save_figure(restDose_DL_rlzs[ii_phantom:ii_phantom+w,jj_phantom:jj_phantom+w], 'DL-{}_{}rlz_01_Mammo_R_CC'.format(typ, rlz))
                    print(restDose_DL_rlzs[ii_phantom:ii_phantom+w,jj_phantom:jj_phantom+w].min(),restDose_DL_rlzs[ii_phantom:ii_phantom+w,jj_phantom:jj_phantom+w].max())
                    break
                            
