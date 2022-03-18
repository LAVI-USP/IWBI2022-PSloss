#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 13:38:48 2022

@author: Rodrigo
"""

import torch

class PowerSpectrum(torch.nn.Module):
    
    '''
    
    Description: Calculates the digital power spectrum (PS).
    
    Input:
        - pixelSize = pixel/detector size
    
    Output:
        - nps1D = 1D PS (radial)
        
    '''
    
    def __init__(self, pixelSize, roiSize, device):
        super(PowerSpectrum, self).__init__()
        
        self.pixelSize = pixelSize
        self.device = device
        
        # NPS 1D - RADIAL - Euclidean Distance
        cx = roiSize//2
        
        self.nFreqSample = cx + 1
        self.nyquist = 1/(2*self.pixelSize)
        
        # Distance matrix (u, v) plane
        x = torch.arange(-cx,roiSize-cx,dtype=torch.float64)
        xx, yy = torch.meshgrid(x,x)  
        self.radialDst = torch.round(torch.sqrt(xx**2 + yy**2))
        
    def forward(self, batch):
             
        # Remove ROI mean
        batch_nm = batch - torch.mean(batch, axis=(2,3), keepdims=True) 
        
        # NPS 2D
        nps2D, _ = self.calc_digital_ps(batch_nm, 2, self.pixelSize)
        
     
        # Generate 1D NPS
        nps1D = torch.empty(self.nFreqSample, device=self.device)
        for k in range(self.nFreqSample):
            nps1D[k] = nps2D[self.radialDst == k].mean()
        
        f1D = torch.linspace(0, self.nyquist, self.nFreqSample) 
        
        return nps1D, nps2D, f1D

    def calc_digital_ps(self, I, n, px = 1):
        '''
        
        Description: Calculates the digital power spectrum (PS) realizations.
        
        Input:
            - I = stack of ROIs
            - n = n-dimensional noise realizations, e.g. 2
            - px = pixel/detector size
            - use_window = Useful for avoiding spectral leakage?
            - average_stack = mean on all ROIs?
            - use_mean = subtract mean or not?
        
        Output:
            - nps = noise-power spectrum (NPS)
            - f = frequency vector
                
        
        -----------------
        
        
        '''
        
        size_I = I.shape
                
        roi_size = size_I[-1]
        
        # Cartesian coordinates
        x = torch.linspace(-roi_size / 2, roi_size / 2, roi_size)
        _, x = torch.meshgrid(x,x)
        
        # frequency vector
        f = torch.linspace(-0.5, 0.5, roi_size) / px
        
        # radial coordinates
        r = torch.sqrt(x**2 + torch.transpose(x,0,1)**2)
        
        # Hann window to avoid spectral leakage
        hann = 0.5 * (1 + torch.cos(3.141592653589793 * r / (roi_size / 2)))
        hann[r > roi_size / 2] = 0
        hann = torch.unsqueeze(torch.unsqueeze(hann, dim=0), dim=0)
        hann = hann.to(self.device)

        F = I * hann
            
        # equivalent to fftn
        F = torch.fft.fftshift(torch.fft.fft2(F, dim=(2,3))) 
        
        # PS
        ps = torch.abs(F) ** 2  
    
        # averaging the NPS over the ROIs assuming ergodicity
        ps = torch.mean(ps, axis=0)
            
        ps = ((px**2)/(roi_size**2)) * ps
        
        ps = torch.squeeze(ps)
                
        return ps, f
    

def pl4_loss(features_y, features_x):
    return torch.mean((features_y.relu4_3 - features_x.relu4_3)**2)
    
def pl3_loss(features_y, features_x):
    return torch.mean((features_y.relu3_3 - features_x.relu3_3)**2)

def MNAE_2D(ps_data, ps_target, mask2D, mask1D):
    return torch.mean(torch.abs(ps_data[1][mask2D] - ps_target[1][mask2D])/ps_target[1][mask2D])

def MNAE_1D(ps_data, ps_target, mask2D, mask1D):
    return torch.mean(torch.abs(ps_data[0][mask1D:]-ps_target[0][mask1D:])/ps_target[0][mask1D:])

