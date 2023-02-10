# -*- coding: utf-8 -*-

import torch.nn as nn

class BidirectionalLSTM(nn.Module):

    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()

        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)

        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)

        return output

class lstmHead(nn.Module):
    def __init__(self,headnum,inchannel,nh,nclass):
        super(lstmHead,self).__init__()
        headlist = nn.ModuleList()
        if headnum == 1:
            headlist.append(BidirectionalLSTM(inchannel, nh, nclass))
        else:
            for i in range(headnum-1):
                if i==0:
                    headlist.append(BidirectionalLSTM(inchannel,nh,nh))
                else:
                    headlist.append(BidirectionalLSTM(nh,nh,nh))
            headlist.append(BidirectionalLSTM(nh,nh,nclass))
        self.headlist = headlist
    def forward(self,x):
        out = []
        for m in self.headlist:
            x= m(x)
            out.append(x)
        return out

# model = lstmHead(4,512,128,64)
# import torch
# data = torch.rand((70,64,512))
# print(model)
# print(model(data)[-1].shape)