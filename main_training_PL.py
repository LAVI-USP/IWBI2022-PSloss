#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 13:38:48 2022

@author: Rodrigo
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import time
import os
import argparse


from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Own codes
from libs.models import ResNetModified, Vgg16
from libs.utilities import load_model, image_grid, makedir
from libs.dataset import BreastCancerDataset
from libs.losses import PowerSpectrum

import libs.pytorch_ssim

#%%

def train(model, 
          vgg, 
          PS,
          func_plx_loss,
          func_ps_loss,
          lambda_plx,
          lambda_ps,
          mask1D,
          optimizer, 
          epoch, 
          train_loader, 
          device, 
          summarywriter):

    # Enable trainning
    model.train()
    
    # create PS mask
    mask2D = (PS.radialDst.numpy() >= mask1D) & (PS.radialDst.numpy() <= 32)

    for step, (data, target) in enumerate(tqdm(train_loader)):

        data = data.to(device)
        target = target.to(device)
                
        # Zero all grads            
        optimizer.zero_grad()
            
        # Generate a batch of new images
        clean_data = model(data)     
      
        # PL
        features_y = vgg(clean_data)
        features_x = vgg(target)
        
        plx_loss = func_plx_loss(features_y, features_x)
        
        # PS loss
        ps_data = PS(clean_data)
        ps_target = PS(target)
        
        loss_ps = func_ps_loss(ps_data, ps_target, mask2D, mask1D)
        
        # Combination of losses
        loss = lambda_plx * plx_loss + lambda_ps * loss_ps
        
        ### Backpropagation ###
        # Calculate all grads
        loss.backward()
        
        # Update weights and biases based on the calc grads 
        optimizer.step()
        
        # ---------------------
        
        # Write model Loss to tensorboard
        summarywriter.add_scalar('Loss/train', 
                                 loss.item(), 
                                 epoch * len(train_loader) + step)
        
        summarywriter.add_scalar('Loss/PS', 
                                 loss_ps.item(), 
                                 epoch * len(train_loader) + step)
        
        summarywriter.add_scalar('Loss/PL', 
                                  plx_loss.item(), 
                                  epoch * len(train_loader) + step)
        
        # Print images to tensorboard
        if step % 20 == 0:
            summarywriter.add_figure('Plot/train', 
                                     image_grid(data[0,0,5:-5,5:-5], 
                                                target[0,0,5:-5,5:-5], 
                                                clean_data[0,0,5:-5,5:-5]),
                                     epoch * len(train_loader) + step,
                                     close=True)
            # Write Gen SSIM to tensorboard
            summarywriter.add_scalar('SSIM/train', 
                                     ssim(clean_data, target).item(), 
                                     epoch * len(train_loader) + step)
        
#%%

if __name__ == '__main__':
    
    ap = argparse.ArgumentParser(description='Restore low-dose mamography')
    ap.add_argument("--rlz", type=int, required=True, 
                    help="Realization number")
    ap.add_argument("--plf", type=str, required=True, 
                    help="Perceptual loss function block")
    ap.add_argument("--psf", type=str, required=True, 
                    help="Power Spectrum loss function type")
    ap.add_argument("--plf_lamb", type=float, required=True, 
                    help="Perceptual loss function weight")
    ap.add_argument("--psf_lamb", type=float, required=True, 
                    help="Power Spectrum loss function weight")
    ap.add_argument("--mask", type=int, required=True, 
                    help="Power Spectrum mask")
    ap.add_argument("--model", type=str, required=True, 
                    help="Model name")

    
    args = vars(ap.parse_args())
    
    rlz = args['rlz'] 
    lambda_plx = args['plf_lamb']
    lambda_ps = args['psf_lamb']
    mask1D = args['mask']
    func_plx_loss = getattr(libs.losses, args['plf'])
    func_ps_loss = getattr(libs.losses, args['psf'])
    model_name = args['model']

    python_path = ""
    
    # Noise scale factor
    mAsFullDose = 60
    mAsLowDose = 30
    
    red_factor = mAsLowDose / mAsFullDose
    
    path_data = "data/"
    path_models = "final_models/rlz_{}/{}/".format(rlz,model_name)
    path_logs = "final_logs/rlz_{}/{}/{}-{}mAs".format(rlz,model_name,time.strftime("%Y-%m-%d-%H%M%S", time.localtime()), mAsLowDose)
    
    path_final_model = path_models + "HResNet_PS-{}mAs.pth".format(mAsLowDose)
    
    LR = 1e-4/10
    batch_size = 128
    n_epochs = 1
    
    dataset_path = '{}DBT_training_{}mAs.h5'.format(path_data,mAsLowDose)
    
    # Tensorboard writer
    summarywriter = SummaryWriter(log_dir=path_logs)
    
    makedir(path_models)
    makedir(path_logs)
    
    # Test if there is a GPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # Create models
    model = ResNetModified()
    
    # Create the optimizer and the LR scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.5, 0.999))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 30, 40, 50], gamma=0.5)    
    
    # Send it to device (GPU if exist)
    model = model.to(device)
    
    # Load gen pre-trained model parameters (if exist)
    start_epoch = load_model(model, 
                             optimizer, 
                             scheduler,
                             path_final_model=path_final_model,
                             path_pretrained_model="final_models/rlz_1/L1/HResNet_PS-{}mAs.pth".format(mAsLowDose))

    # Create dataset helper
    train_set = BreastCancerDataset(dataset_path, red_factor, vmin=48., vmax=2000.)
    
    # Create dataset loader
    train_loader = torch.utils.data.DataLoader(train_set,
                                              batch_size=batch_size, 
                                              shuffle=True,
                                              pin_memory=True)
    
    ssim = libs.pytorch_ssim.SSIM(window_size = 11)

    vgg = Vgg16(requires_grad=False).to(device)
    
    PS = PowerSpectrum(pixelSize=0.14, roiSize=64, device=device)
        
    # Loop on epochs
    for epoch in range(start_epoch, n_epochs):
        
      print("Epoch:[{}] LR:{}".format(epoch, optimizer.state_dict()['param_groups'][0]['lr']))
    
      # Train the model for 1 epoch
      train(model,
               vgg,
               PS,
               func_plx_loss,
               func_ps_loss,
               lambda_plx,
               lambda_ps,
               mask1D,
               optimizer,
               epoch,
               train_loader,
               device,
               summarywriter)

      # Update LR
      scheduler.step()
    
      # Save the model
      torch.save({
                 'epoch': epoch,
                 'model_state_dict': model.state_dict(),
                 'optimizer_state_dict': optimizer.state_dict(),
                 'scheduler_state_dict': scheduler.state_dict(),
                 }, path_final_model)
      
      if (epoch + 1) % 1 == 0:
        # Testing code
        model = model.to('cpu')
        torch.cuda.empty_cache()
        os.system("{} main_testing.py --rlz {} --mas_ld {} --typ {}".format(python_path, rlz, mAsLowDose,model_name))
        os.system("{} evaluation/PS_rlz.py --rlz {} --mas_ld {} --typ {}".format(python_path, rlz, mAsLowDose,model_name))
        os.system("{} evaluation/MNSE.py".format(python_path))
        model = model.to(device)
