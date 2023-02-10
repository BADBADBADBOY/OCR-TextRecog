# -*- coding: utf-8 -*-

from torch.nn import CTCLoss as PytorchCTCLoss
# from warpctc_pytorch import CTCLoss as WarpCTCLoss
import torch.nn as nn


# class CTCLoss(nn.Module):
#     def __init__(self, ):
#         super(CTCLoss, self).__init__()
#         self.criterion = WarpCTCLoss()
#     def forward(self, preds, labels, preds_size, labels_len):
#         ## preds, text, preds_size, length
#         loss = self.criterion(preds, labels, preds_size, labels_len)
#         return loss


class CTCLoss(nn.Module):
    def __init__(self, ):
        super(CTCLoss, self).__init__()
        self.criterion = PytorchCTCLoss(reduction='none')
    def forward(self, preds, labels, preds_size, labels_len):
        ## preds, text, preds_size, length
        preds = preds.log_softmax(2).requires_grad_()  # torch.ctcloss
        loss = self.criterion(preds, labels, preds_size, labels_len)
        loss = loss.sum()
        return loss