# In src/vgg16.py

import torch
import torch.nn as nn
from torchvision.models import vgg16, VGG16_Weights

class VGG16(nn.Module):
    def __init__(self, requires_grad=False):
        super().__init__()
        # Use the recommended modern weights API
        vgg_pretrained_features = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features
        
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()

        # conv1_1, conv1_2
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        # conv2_1, conv2_2
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        # conv3_1, conv3_2, conv3_3
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        # conv4_1, conv4_2, conv4_3
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        # Normalize the input
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(x.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(x.device)
        x = (x - mean) / std
        
        h_relu1_2 = self.slice1(x)
        h_relu2_2 = self.slice2(h_relu1_2)
        h_relu3_3 = self.slice3(h_relu2_2)
        h_relu4_3 = self.slice4(h_relu3_3)
        
        out = {
            "conv1_2": h_relu1_2,
            "conv2_2": h_relu2_2,
            "conv3_3": h_relu3_3,
            "conv4_3": h_relu4_3,
        }
        return out