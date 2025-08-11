import torch
import torch.nn as nn
from torchvision.models import vgg16

class VGG16(nn.Module):
    def __init__(self, requires_grad=False):
        super(VGG16, self).__init__()
        vgg_features = vgg16(pretrained=True).features
        
        # Correctly map layer names to slices of the VGG network
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        self.slice5 = nn.Sequential()
        
        for x in range(4):
            self.slice1.add_module(str(x), vgg_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_features[x])
        for x in range(23, 30):
            self.slice5.add_module(str(x), vgg_features[x])
            
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_conv1_2 = h
        h = self.slice2(h)
        h_conv2_2 = h
        h = self.slice3(h)
        h_conv3_3 = h
        h = self.slice4(h)
        h_conv4_3 = h
        h = self.slice5(h)
        h_conv5_3 = h
        
        # Return features from the same layers as the original paper
        return {
            'conv1_2': h_conv1_2,
            'conv2_2': h_conv2_2,
            'conv3_3': h_conv3_3,
            'conv4_3': h_conv4_3,
            'conv5_3': h_conv5_3,
        }