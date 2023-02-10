import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np

class Im2Seq(nn.Module):
    def __init__(self, in_channels, **kwargs):
        super().__init__()
        self.out_channels = in_channels

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == 1
        x = x.squeeze(axis=2)
        x = x.permute([0, 2, 1])  # (NTC)(batch, width, channels)
        return x

class CTCHead(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 mid_channels=None,
                 return_feats=False,
                 **kwargs):
        super(CTCHead, self).__init__()
        if mid_channels is None:
            self.fc = nn.Linear(
                in_channels,
                out_channels)
        else:
            self.fc1 = nn.Linear(
                in_channels,
                mid_channels)
            self.fc2 = nn.Linear(
                mid_channels,
                out_channels)
        self.out_channels = out_channels
        self.mid_channels = mid_channels
        self.return_feats = return_feats

    def forward(self, x, targets=None):
        if self.mid_channels is None:
            predicts = self.fc(x)
        else:
            x = self.fc1(x)
            predicts = self.fc2(x)

        if self.return_feats:
            result = (x, predicts)
        else:
            result = predicts
        return result
    
class svtrHead(nn.Module):
    def __init__(self, in_channels,out_channels,mid_channels,return_feats, **kwargs):
        super().__init__()
        self.neck = Im2Seq(in_channels)
        self.head = CTCHead(in_channels,out_channels,mid_channels,return_feats)
        
    def forward(self,x):
        x = self.neck(x)
        x = self.head(x)
        return x