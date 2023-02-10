# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

class CRNNKLDivLoss(nn.Module):
    def __init__(self,reduction='batchmean'):
        super(CRNNKLDivLoss,self).__init__()
        self.reduction = reduction
    def forward(self,input,target):
        input = F.log_softmax(input,-1)
        target = F.softmax(target,-1)
        return F.kl_div(input, target, reduction=self.reduction)