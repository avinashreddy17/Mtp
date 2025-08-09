# In src/modules.py

import torch.nn as nn
import torch.nn.functional as F

def conv_block(in_channels, out_channels, kernel_size, stride=1, padding=0, activation='relu', use_batchnorm=False):
    """
    Creates a convolutional block.
    """
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
    ]
    
    if use_batchnorm:
        layers.append(nn.BatchNorm2d(out_channels))
    
    if activation == 'relu':
        layers.append(nn.ReLU(inplace=True))
    elif activation == 'leaky_relu':
        layers.append(nn.LeakyReLU(0.2, inplace=True))
    elif activation == 'tanh':
        layers.append(nn.Tanh())
        
    return nn.Sequential(*layers)

def deconv_block(in_channels, out_channels, kernel_size, stride=1, padding=0, activation='relu', use_batchnorm=False):
    """
    Creates a transposed convolutional block.
    """
    layers = [
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding)
    ]
    
    if use_batchnorm:
        layers.append(nn.BatchNorm2d(out_channels))
        
    if activation == 'relu':
        layers.append(nn.ReLU(inplace=True))
        
    return nn.Sequential(*layers)

def linear_block(in_features, out_features, activation='relu', use_batchnorm=False):
    """
    Creates a linear block.
    """
    layers = [
        nn.Linear(in_features, out_features)
    ]

    if use_batchnorm:
        # BatchNorm for linear layers is 1D
        layers.append(nn.BatchNorm1d(out_features))

    if activation == 'relu':
        layers.append(nn.ReLU(inplace=True))
    elif activation == 'leaky_relu':
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        
    return nn.Sequential(*layers)