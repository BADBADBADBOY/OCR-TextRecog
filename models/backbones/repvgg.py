# -*- coding: utf-8 -*-

import torch.nn as nn
import torch
from .repvggblock import RepVGGBlock,repvgg_model_convert

class RepVGG(nn.Module):

    def __init__(self, num_blocks, in_channels=3, width_multiplier=None, override_groups_map=None, deploy=False,
                 use_se=False):
        super(RepVGG, self).__init__()

        assert len(width_multiplier) == 4

        self.deploy = deploy
        self.override_groups_map = override_groups_map or dict()
        self.use_se = use_se
        stride = [1, 2, 2, 2]

        assert 0 not in self.override_groups_map

        self.in_planes = min(64, int(64 * width_multiplier[0]))

        self.stage0 = RepVGGBlock(in_channels=in_channels, out_channels=self.in_planes, kernel_size=3, stride=2, padding=1,
                                  deploy=self.deploy, use_se=self.use_se)
        self.cur_layer_idx = 1
        self.stage1 = self._make_stage(int(64 * width_multiplier[0]), num_blocks[0], stride=(stride[0], 1))
        self.stage2 = self._make_stage(int(128 * width_multiplier[1]), num_blocks[1], stride=(stride[1], 1))
        self.stage3 = self._make_stage(int(256 * width_multiplier[2]), num_blocks[2], stride=(stride[2], 1))
        self.stage4 = self._make_stage(int(512 * width_multiplier[3]), num_blocks[3], stride=(stride[3], 1))
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

    def _make_stage(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        blocks = []
        for stride in strides:
            cur_groups = self.override_groups_map.get(self.cur_layer_idx, 1)
            blocks.append(RepVGGBlock(in_channels=self.in_planes, out_channels=planes, kernel_size=3,
                                      stride=stride, padding=1, groups=cur_groups, deploy=self.deploy,
                                      use_se=self.use_se))
            self.in_planes = planes
            self.cur_layer_idx += 1
        self.out_channels = self.in_planes
        return nn.Sequential(*blocks)

    def forward(self, x):
        out = self.stage0(x)
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)
        out = self.pool(out)
        return out


optional_groupwise_layers = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26]
g2_map = {l: 2 for l in optional_groupwise_layers}
g4_map = {l: 4 for l in optional_groupwise_layers}

def create_RepVGG_Small(in_channels =3,deploy=False):
    return RepVGG(num_blocks=[2, 4, 8, 1], in_channels=in_channels,
                  width_multiplier=[1, 0.75, 0.5, 1], override_groups_map=None, deploy=deploy)

def create_RepVGG_A(in_channels =3,deploy=False):
    return RepVGG(num_blocks=[2, 4, 14, 1], in_channels=in_channels,
                  width_multiplier=[1, 0.75, 0.5, 2], override_groups_map=None, deploy=deploy)

def create_RepVGG_A0(in_channels =3,deploy=False):
    return RepVGG(num_blocks=[2, 4, 14, 1], in_channels=in_channels,
                  width_multiplier=[0.75, 0.75, 0.75, 2.5], override_groups_map=None, deploy=deploy)


def create_RepVGG_A1(in_channels=3,deploy=False):
    return RepVGG(num_blocks=[2, 4, 14, 1], in_channels=in_channels,
                  width_multiplier=[1, 1, 1, 2.5], override_groups_map=None, deploy=deploy)


def create_RepVGG_A2(in_channels =3,deploy=False):
    return RepVGG(num_blocks=[2, 4, 14, 1], in_channels=in_channels,
                  width_multiplier=[1.5, 1.5, 1.5, 2.75], override_groups_map=None, deploy=deploy)


def create_RepVGG_B0(in_channels =3,deploy=False):
    return RepVGG(num_blocks=[4, 6, 16, 1], in_channels=in_channels,
                  width_multiplier=[1, 1, 1, 2.5], override_groups_map=None, deploy=deploy)


def create_RepVGG_B1(in_channels =3,deploy=False):
    return RepVGG(num_blocks=[4, 6, 16, 1], in_channels=in_channels,
                  width_multiplier=[2, 2, 2, 4], override_groups_map=None, deploy=deploy)


def create_RepVGG_B1g2(in_channels =3,deploy=False):
    return RepVGG(num_blocks=[4, 6, 16, 1], in_channels=in_channels,
                  width_multiplier=[2, 2, 2, 4], override_groups_map=g2_map, deploy=deploy)


def create_RepVGG_B1g4(in_channels =3,deploy=False):
    return RepVGG(num_blocks=[4, 6, 16, 1], in_channels=in_channels,
                  width_multiplier=[2, 2, 2, 4], override_groups_map=g4_map, deploy=deploy)


def create_RepVGG_B2(in_channels =3,deploy=False):
    return RepVGG(num_blocks=[4, 6, 16, 1], in_channels=in_channels,
                  width_multiplier=[2.5, 2.5, 2.5, 5], override_groups_map=None, deploy=deploy)


def create_RepVGG_B2g2(in_channels =3,deploy=False):
    return RepVGG(num_blocks=[4, 6, 16, 1], in_channels=in_channels,
                  width_multiplier=[2.5, 2.5, 2.5, 5], override_groups_map=g2_map, deploy=deploy)


def create_RepVGG_B2g4(in_channels =3,deploy=False):
    return RepVGG(num_blocks=[4, 6, 16, 1], in_channels=in_channels,
                  width_multiplier=[2.5, 2.5, 2.5, 5], override_groups_map=g4_map, deploy=deploy)


def create_RepVGG_B3(in_channels =3,deploy=False):
    return RepVGG(num_blocks=[4, 6, 16, 1],in_channels=in_channels,
                  width_multiplier=[3, 3, 3, 5], override_groups_map=None, deploy=deploy)


def create_RepVGG_B3g2(in_channels =3,deploy=False):
    return RepVGG(num_blocks=[4, 6, 16, 1], in_channels=in_channels,
                  width_multiplier=[3, 3, 3, 5], override_groups_map=g2_map, deploy=deploy)


def create_RepVGG_B3g4(in_channels =3,deploy=False):
    return RepVGG(num_blocks=[4, 6, 16, 1], in_channels=in_channels,
                  width_multiplier=[3, 3, 3, 5], override_groups_map=g4_map, deploy=deploy)


def create_RepVGG_D2se(in_channels =3,deploy=False):
    return RepVGG(num_blocks=[8, 14, 24, 1], in_channels=in_channels,
                  width_multiplier=[2.5, 2.5, 2.5, 5], override_groups_map=None, deploy=deploy, use_se=True)


func_dict = {
    'RepVGG-Small':create_RepVGG_Small,
    'RepVGG-A': create_RepVGG_A,
    'RepVGG-A0': create_RepVGG_A0,
    'RepVGG-A1': create_RepVGG_A1,
    'RepVGG-A2': create_RepVGG_A2,
    'RepVGG-B0': create_RepVGG_B0,
    'RepVGG-B1': create_RepVGG_B1,
    'RepVGG-B1g2': create_RepVGG_B1g2,
    'RepVGG-B1g4': create_RepVGG_B1g4,
    'RepVGG-B2': create_RepVGG_B2,
    'RepVGG-B2g2': create_RepVGG_B2g2,
    'RepVGG-B2g4': create_RepVGG_B2g4,
    'RepVGG-B3': create_RepVGG_B3,
    'RepVGG-B3g2': create_RepVGG_B3g2,
    'RepVGG-B3g4': create_RepVGG_B3g4,
    'RepVGG-D2se': create_RepVGG_D2se,  # Updated at April 25, 2021. This is not reported in the CVPR paper.
}

def get_RepVGG_func_by_name(name):
    return func_dict[name]

def CreateBackboneModel(modelName,pretrained=False,scale=1,isGray=False):
    assert modelName in ['RepVGG-Small','RepVGG-A','RepVGG-A0','RepVGG-A1','RepVGG-A2','RepVGG-B0','RepVGG-B1','RepVGG-B1g2','RepVGG-B1g4','RepVGG-B2','RepVGG-B2g2','RepVGG-B2g4','RepVGG-B3','RepVGG-B3g2','RepVGG-B3g4','RepVGG-D2se']
    in_channels=1 if isGray else 3
    model = get_RepVGG_func_by_name(modelName)(in_channels)
    return model

# model,mapChannels = CreateBackboneModel('RepVGG-B2',pretrained=False,scale=1,isGray=True)
# print(model,mapChannels)
# img = torch.rand((1,1,32,280))
# out = model(img)
# print(out.shape)

