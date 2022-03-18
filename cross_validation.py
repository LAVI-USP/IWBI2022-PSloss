#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 13:38:48 2022

@author: Rodrigo
"""

import os

python_path = ""

modelnames = ['pl4_loss', 'pl3_loss', 'MNAE_2D', 'MNAE_1D']
plfs = ['pl4_loss', 'pl3_loss', 'pl4_loss', 'pl4_loss']
psfs = ['MNAE_2D', 'MNAE_2D', 'MNAE_2D', 'MNAE_1D']
plf_lambs = [1, 1, 0, 0]
psf_lambs = [0, 0, 1, 1]
mask = 0

for rlz in range(1,11):

    for idX, modelname in enumerate(modelnames):

        print("\n{} - {}\n".format(modelname,rlz))
        file1 = open("outputs.txt","a")
        file1.write("\n{} - {}\n".format(modelname,rlz))
        file1.close()

        python_command = "{} main_training_PL.py \
                  --rlz {} --plf {} --psf {} \
                  --plf_lamb {} --psf_lamb {} --mask {} \
                  --model {}".format(python_path, rlz,
                  plfs[idX], psfs[idX], plf_lambs[idX], psf_lambs[idX], mask, modelname)

        print(python_command)
        os.system(python_command)
        os.system("{} evaluation/save_phantom_png.py --rlz {} --typ {}".format(python_path, rlz, modelname))
        


modelnames = ['Combined']
plfs = ['pl4_loss']
psfs = ['MNAE_2D']
plf_lambs = [1]
psf_lambs = [0.027, 0.023, 0.003, 0.037, 0.212, 0.939, 0.624, 0.044, 0.375, 0.143] 
mask = 0

for rlz in range(1,11):
    
    for idX, modelnameX in enumerate(modelnames):

        for psf_lamb in psf_lambs:

            modelname = modelnameX + '-{}'.format(psf_lamb)

            print("\n{} - {}\n".format(modelname,rlz))
            file1 = open("outputs.txt","a")
            file1.write("\n{} - {}\n".format(modelname,rlz))
            file1.close()

            python_command = "{} main_training_PL.py \
                      --rlz {} --plf {} --psf {} \
                      --plf_lamb {} --psf_lamb {} --mask {} \
                      --model {}".format(python_path, rlz,
                      plfs[idX], psfs[idX], plf_lambs[idX], psf_lamb, mask, modelname)

            print(python_command)
            os.system(python_command)
            os.system("{} evaluation/save_phantom_png.py --rlz {} --typ {}".format(python_path, rlz, modelname))