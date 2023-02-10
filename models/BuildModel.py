# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.nn import functional as F
from utils.tools import create_module
from models.backbones.tps_spatial_transformer import TPSSpatialTransformer
from models.backbones.stn_head import STNHead,repSTNHead

class crnnModel(nn.Module):
    def __init__(self,config):
        super(crnnModel,self).__init__()
        if config['model']['STN']['STN_ON']:
            self.tps = TPSSpatialTransformer(output_image_size=tuple(config['model']['STN']['tps_outputsize']),
                                    num_control_points=config['model']['STN']['num_control_points'],
                                    margins=tuple(config['model']['STN']['tps_margins']))
            if config['model']['STN']['use_rep']:
                self.stn_head = repSTNHead(in_planes=3,num_ctrlpoints=config['model']['STN']['num_control_points'],activation=config['model']['STN']['stn_activation'])
            else:
                self.stn_head = STNHead(in_planes=3,num_ctrlpoints=config['model']['STN']['num_control_points'],activation=config['model']['STN']['stn_activation'])
        
        
        self.backbone = create_module(config['model']['backbone_function'])(config['model']['backbone'],
                                                                   pretrained=config['model']['pretrained'],
                                                                   scale=config['model']['scale'],
                                                                   isGray=config['train']['isGray'])
        self.head = create_module(config['model']['head_function'])(config['model']['headnum'],
                                                                    self.backbone.out_channels,
                                                                    config['model']['nh'],
                                                                    config['train']['nclass'])
        self.config = config

    def forward(self,x):
        if self.config['model']['STN']['STN_ON']:
            stn_input = F.interpolate(x, self.config['model']['STN']['tps_inputsize'], mode='bilinear', align_corners=True)
            stn_img_feat, ctrl_points = self.stn_head(stn_input)
            stn_x,_ = self.tps(x, ctrl_points)
            x = self.backbone(stn_x)
        else:
            x = self.backbone(x)
        if self.config['model']['head_function'].split(',')[-1]=='lstmHead':
            b, c, h, w = x.size()
            assert h == 1, "the height of conv must be 1"
            x = x.squeeze(2)
            x = x.permute(2, 0, 1)
        out = self.head(x)
        map_class,map_fea = out[-1],out[-2]
        if self.config['model']['head_function'].split(',')[-1]=='convHead':
            b, c, h, w = map_class.size()
            assert h == 1, "the height of conv must be 1"
            map_class = map_class.squeeze(2)
            map_class = map_class.permute(2, 0, 1) ## T*C*classes
        if self.training:
            return map_class,map_fea
        else:
            return map_class,map_fea,stn_x


class SVTRModel(nn.Module):
    def __init__(self,config):
        super(SVTRModel,self).__init__()
        if config['model']['STN']['STN_ON']:
            self.tps = TPSSpatialTransformer(output_image_size=tuple(config['model']['STN']['tps_outputsize']),
                                    num_control_points=config['model']['STN']['num_control_points'],
                                    margins=tuple(config['model']['STN']['tps_margins']))
            if config['model']['STN']['use_rep']:
                self.stn_head = repSTNHead(in_planes=3,num_ctrlpoints=config['model']['STN']['num_control_points'],activation=config['model']['STN']['stn_activation'])
            else:
                self.stn_head = STNHead(in_planes=3,num_ctrlpoints=config['model']['STN']['num_control_points'],activation=config['model']['STN']['stn_activation'])
        
        
        self.backbone = create_module(config['model']['backbone_function'])(img_size=config['model']['backbone']['img_size'],
                                                      in_channels=config['model']['backbone']['in_channels'],
                                                      embed_dim=config['model']['backbone']['embed_dim'],
                                                      depth=config['model']['backbone']['depth'],
                                                      num_heads=config['model']['backbone']['num_heads'],
                                                      mixer=eval(config['model']['backbone']['mixer']) 
                                                      if isinstance(config['model']['backbone']['mixer'],str)
                                                      else config['model']['backbone']['mixer'],
                                                      local_mixer=config['model']['backbone']['local_mixer'],
                                                      patch_merging=config['model']['backbone']['patch_merging'],
                                                      out_channels=config['model']['backbone']['out_channels'],
                                                      out_char_num=config['model']['backbone']['out_char_num'],
                                                      last_stage=config['model']['backbone']['last_stage'],
                                                      sub_num=config['model']['backbone']['sub_num'],
                                                      prenorm=config['model']['backbone']['prenorm'],
                                                      use_lenhead=config['model']['backbone']['use_lenhead'],
                                                      local_rank=config['train']['local_rank'])
        
        self.head = create_module(config['model']['head_function'])(in_channels=config['model']['backbone']['out_channels'],
                                                out_channels=config['train']['nclass'],
                                                mid_channels=eval(config['model']['backbone']['mid_channels']),
                                                return_feats=config['model']['backbone']['return_feats'])
        self.config = config

    def forward(self,x):
        if self.config['model']['STN']['STN_ON']:
            stn_input = F.interpolate(x, self.config['model']['STN']['tps_inputsize'], mode='bilinear', align_corners=True)
            stn_img_feat, ctrl_points = self.stn_head(stn_input)
            stn_x,_ = self.tps(x, ctrl_points)
            x = self.backbone(stn_x)
        else:
            x = self.backbone(x)
        out = self.head(x)
        if self.training:
            return out.permute(1, 0, 2),None
        else:
            return out.permute(1, 0, 2),None,stn_x
