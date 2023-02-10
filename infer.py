# -*- coding: utf-8 -*-

import torch
import numpy as np
import os
import cv2
import yaml
import argparse
import shutil
from models.BuildModel import crnnModel,SVTRModel
from utils.label2tensor import strLabelConverter
from models.backbones.repvggblock import repvgg_model_convert
from utils.tools import loadkeyfile,create_module,makedir
from PIL import Image,ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torchvision.transforms as transforms
from utils.tools import tensor2img


class CRNNInfer(object):
    def __init__(self,config):
        self.converter = strLabelConverter(config['train']['alphabet'])
        config['train']['local_rank'] = torch.device('cuda')
        if config['train']['algorithm']=='CRNN':
            model = crnnModel(config).cuda(config['train']['local_rank'])
        elif config['train']['algorithm']=='SVTR':
            model = SVTRModel(config).cuda(config['train']['local_rank'])
        model_dict = torch.load(config['infer']['model_file'])['state_dict']
        new_dict = {}
        for key in model.state_dict().keys():
            new_dict[key] = model_dict['module.'+key]
        model.load_state_dict(new_dict)
#         self.model = repvgg_model_convert(model,None)
        self.model = model
        self.model.eval() 
        self.isGray = config['train']['isGray']
        print(model.state_dict().keys())
        torch.save(model.state_dict(),'model.pth')
    def recogimg(self,img):
        w, h = img.size
#         new_w = int(32 / float(h) * w)//4*4
#         if new_w>128:
#             new_w = 128
#         img = img.resize((new_w, 32), Image.BILINEAR)
        img = img.resize((256, 48), Image.BILINEAR)
        if self.isGray:
            img = img.convert('L')
        img = transforms.ToTensor()(img)
        img.sub_(0.5).div_(0.5)
        with torch.no_grad():
            preds_class,preds_map,stn_x = self.model(img.unsqueeze(0).cuda())
        cv2.imwrite('stn.jpg',tensor2img(stn_x[0]))
        preds_size = torch.IntTensor([preds_class.size(0)] * preds_class.size(1))
        _, preds_class = preds_class.max(2)
        preds_class = preds_class.squeeze(1).contiguous().view(-1)
        sim_preds_char = self.converter.decode(preds_class.data, preds_size.data, raw=False)
        return sim_preds_char
    
class CRNNInferCPU(object):
    def __init__(self,config):
        self.converter = strLabelConverter(config['train']['alphabet'])
        config['train']['local_rank'] = torch.device('cpu')
        if config['train']['algorithm']=='CRNN':
            model = crnnModel(config).to(config['train']['local_rank'])
        elif config['train']['algorithm']=='SVTR':
            model = SVTRModel(config).to(config['train']['local_rank'])
        import pdb
        pdb.set_trace()
        model_dict = torch.load(config['infer']['model_file'],map_location='cpu')['state_dict']
        new_dict = {}
        for key in model.state_dict().keys():
            new_dict[key] = model_dict['module.'+key]
        model.load_state_dict(new_dict)
#         self.model = repvgg_model_convert(model,None)
        self.model.eval() 
        self.isGray = config['train']['isGray']
    def recogimg(self,img):
        import time
        t = time.time()
        w, h = img.size
#         new_w = int(32 / float(h) * w)//4*4
#         if new_w>128:
#             new_w = 128
#         img = img.resize((new_w, 32), Image.BILINEAR)
        img = img.resize((256, 48), Image.BILINEAR)
        if self.isGray:
            img = img.convert('L')
        img = transforms.ToTensor()(img)
        img.sub_(0.5).div_(0.5)
        print('dataTime:{} s'.format(time.time() - t))
        t = time.time()
        with torch.no_grad():
            preds_class,preds_map = self.model(img.unsqueeze(0))
        print('modelTime:{} s'.format(time.time() - t))
        t = time.time()
        preds_size = torch.IntTensor([preds_class.size(0)] * preds_class.size(1))
        _, preds_class = preds_class.max(2)
        preds_class = preds_class.squeeze(1).contiguous().view(-1)
        sim_preds_char = self.converter.decode(preds_class.data, preds_size.data, raw=False)
        print('decodeTime:{} s'.format(time.time() - t))
        return sim_preds_char
        

def demo(opt):
    path = opt.path
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpuid
    config = yaml.load(open(opt.config,'r',encoding='utf-8'),Loader=yaml.FullLoader)
    config['train']['alphabet'] = loadkeyfile(config['train']['key_file'])
    config['train']['nclass'] = len(config['train']['alphabet']) + 1
    recog_nin = CRNNInfer(config)
#     recog_nin = CRNNInferCPU(config)
    if os.path.isdir(path):
        for file in os.listdir(path):
            print(os.path.join(path,file))
            img = Image.open(os.path.join(path,file)).convert('RGB')
            sim_preds_char = recog_nin.recogimg(img)
            print(sim_preds_char)
            shutil.copy(os.path.join(path,file),'./img.jpg')
#             break
    else:
        img = Image.open(path).convert('RGB')
        sim_preds_char = recog_nin.recogimg(img)
        print(sim_preds_char)
            
        
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='./11.jpg', help='DDP parameter, do not modify')
    parser.add_argument('--gpuid', type=str, default='2', help='DDP parameter, do not modify')
    parser.add_argument('--config', type=str, default='./config/SVTR-T-Ch-repstn.yaml', help='DDP parameter, do not modify')
    opt = parser.parse_args()
    demo(opt)