# -*- coding: utf-8 -*-
import torch.nn as TN

class conv_bn_relu(TN.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels,
                 kernel_size=1,
                 padding=0,
                 stride=1, 
                 eps=1e-5, 
                 momentum=0.1,
                 with_bn=True):
        super().__init__()

        if with_bn:
            self.conv_bn_relu = TN.Sequential(TN.Conv2d(in_channels=in_channels,
                                                        out_channels=out_channels,
                                                        kernel_size=kernel_size,
                                                        padding=padding,
                                                        stride=stride,
                                                        bias=True),
                                              TN.BatchNorm2d(num_features=out_channels,
                                                             eps=eps,
                                                             momentum=momentum),
                                              TN.ReLU())
        else:
            self.conv_bn_relu = TN.Sequential(TN.Conv2d(in_channels=in_channels,
                                                        out_channels=out_channels,
                                                        kernel_size=kernel_size,
                                                        padding=padding,
                                                        stride=stride,
                                                        bias=True),
                                              TN.ReLU())
            
        for m in self.modules():
            if isinstance(m, TN.Conv2d):
                TN.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    TN.init.constant_(m.bias,0.1)
            elif isinstance(m, TN.BatchNorm2d):
                if m.weight is not None:
                    TN.init.constant_(m.weight, 1)
                    TN.init.constant_(m.bias, 0)
    
    def forward(self,x):
        x=self.conv_bn_relu(x)
        return x
    
class Flatten(TN.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x=x.view(x.size(0),-1)
        return x