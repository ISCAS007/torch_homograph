# -*- coding: utf-8 -*-

from utils.torch_layers import conv_bn_relu, Flatten
import torch.nn as nn

class vggstyle_homo(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.args=args
        self.convs=nn.Sequential(conv_bn_relu(in_channels=2,out_channels=64),
                                 conv_bn_relu(in_channels=64,out_channels=64),
                                 nn.MaxPool2d(kernel_size=2,stride=2),
                                 conv_bn_relu(in_channels=64,out_channels=64),
                                 conv_bn_relu(in_channels=64,out_channels=64),
                                 nn.MaxPool2d(kernel_size=2,stride=2),
                                 conv_bn_relu(in_channels=64,out_channels=128),
                                 conv_bn_relu(in_channels=128,out_channels=128),
                                 nn.MaxPool2d(kernel_size=2,stride=2),
                                 conv_bn_relu(in_channels=128,out_channels=128),
                                 conv_bn_relu(in_channels=128,out_channels=128),
                                 Flatten(),
                                 nn.Dropout2d(p=0.5),
                                 nn.Linear(in_features=16*16*128,out_features=1024,bias=True),
                                 nn.Dropout2d(p=0.5),
                                 nn.Linear(in_features=1024,out_features=8,bias=True))
        
        # use the default init from pytorch for nn.Linear
    
    def forward(self,x):
        x=self.convs(x)
        return x
                                 