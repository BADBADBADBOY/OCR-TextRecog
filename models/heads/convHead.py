# -*- coding: utf-8 -*-

import torch.nn as nn

class convHead(nn.Module):
    def __init__(self,headnum,inchannel,nh,nclass):
        super(convHead,self).__init__()
        self.downconv = nn.Sequential(nn.Conv2d(inchannel,nh,1,1,0),nn.BatchNorm2d(nh),nn.ReLU())
        self.concClass = nn.Conv2d(nh,nclass,1,1,0)
    def forward(self,x):
        out = []
        x = self.downconv(x)
        out.append(x)
        x = self.concClass(x)
        out.append(x)
        return out
