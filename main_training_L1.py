#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 13:38:48 2022

@author: Rodrigo
"""

import matplotlib.pyplot as plt
import torch
import time
import os
import argparse

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Own codes
from libs.models import ResNetModified
from libs.utilities import load_model, image_grid, makedir
from libs.dataset import BreastCancerDataset

import libs.pytorch_ssim

#%%

def train(model, optimizer, epoch, train_loader, device, summarywriter):

    # Enable trainning
    model.train()

    for step, (data, target) in enumerate(tqdm(train_loader)):

        data = data.to(device)
        target = target.to(device)
                
        # Zero all grads            
        optimizer.zero_grad()
            
        # Generate a batch of new images
        clean_data = model(data)
      
        # L1 loss
        loss = torch.mean(torch.abs(clean_data - target))

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
        
        # Print images to tensorboard
        if step % 20 == 0:
            summarywriter.add_figure('Plot/train', 
                                     image_grid(data[0,0,:,:], 
                                                target[0,0,:,:], 
                                                clean_data[0,0,:,:]),
                                     epoch * len(train_loader) + step,
                                     close=True)
            # Write Gen SSIM to tensorboard
            summarywriter.add_scalar('SSIM/train', 
                                     ssim(clean_data, target).item(), 
                                     epoch * len(train_loader) + step)
        

#%%

if __name__ == '__main__':
    
    rlz = 1
        
    # Noise scale factor
    mAsFullDose = 60
    mAsLowDose = 30
    
    red_factor = mAsLowDose / mAsFullDose
    
    path_data = "data/"
    path_models = "final_models/rlz_{}/L1/".format(rlz)
    path_logs = "final_logs/rlz_{}/{}-{}mAs".format(rlz,time.strftime("%Y-%m-%d-%H%M%S", time.localtime()), mAsLowDose)
    
    path_final_model = path_models + "HResNet_PS-{}mAs.pth".format(mAsLowDose)
    
    LR = 1e-3
    batch_size = 256
    n_epochs = 60
    
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
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 30, 40, 50], gamma=0.5)    
    
    # Send it to device (GPU if exist)
    model = model.to(device)
    
    # Load gen pre-trained model parameters (if exist)
    start_epoch = load_model(model, 
                             optimizer, 
                             scheduler,
                             path_final_model=path_final_model,
                             path_pretrained_model=path_final_model)
    
    # Create dataset helper
    train_set = BreastCancerDataset(dataset_path, vmin=48., vmax=2000., red_factor=red_factor)
    
    # Create dataset loader
    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=batch_size, 
                                               shuffle=True,
                                               pin_memory=True)
    
    ssim = libs.pytorch_ssim.SSIM(window_size = 11)
        
    # Loop on epochs
    for epoch in range(start_epoch, n_epochs):
        
      print("Epoch:[{}] LR:{}".format(epoch, optimizer.state_dict()['param_groups'][0]['lr']))
    
      # Train the model for 1 epoch
      train(model,
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
      
      if (epoch + 1) % 10 == 0:
          # Testing code
          os.system("python main_testing.py --rlz {} --mas_ld {} --typ L1".format(rlz, mAsLowDose))
          exec(open("evaluation/MNSE.py").read())