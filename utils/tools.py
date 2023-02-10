
import os
import torch
import importlib
from torch.autograd import Variable
import numpy as np

def get_no_weight_decay_param(model,config):
    param_names = config['optimizer']['no_weight_decay_param']['param_names']
    weight_decay = config['optimizer']['no_weight_decay_param']['weight_decay']
    is_on = config['optimizer']['no_weight_decay_param']['is_ON']
    if not is_on:
        return model.parameters()
    base_param = []
    no_weight_decay_param = []
    for (name,param) in model.named_parameters():
        is_no_weight = False
        for param_name in param_names:
            if param_name in name:
                is_no_weight = True
                break      
        if is_no_weight:
            no_weight_decay_param.append(param)
        else:
            base_param.append(param)      
    Outparam = [{'params': base_param},{'params': no_weight_decay_param, 'weight_decay': weight_decay}]
    return Outparam

def fix_param(model,config):
    param_names = config['optimizer']['no_weight_decay_param']['param_names']
    weight_decay = config['optimizer']['no_weight_decay_param']['weight_decay']
    is_on = config['optimizer']['no_weight_decay_param']['is_ON']
    STN_ON = config['model']['STN']['STN_ON']
    stn_lr = config['model']['STN']['stn_lr']
    
    base_param = []
    stn_param = []
    no_weight_decay_param = []
    for (name,param) in model.named_parameters():
        is_no_weight = False
        for param_name in param_names:
            if param_name in name:
                is_no_weight = True
                break      
        if is_no_weight:
            no_weight_decay_param.append(param)
        elif 'stn' in name:
            stn_param.append(param)
        else:
            base_param.append(param)
    Outparam = [{'params': base_param},{'params': stn_param},{'params': no_weight_decay_param}]
    
    if STN_ON:
        Outparam[1]['lr'] = stn_lr
    if is_on:
        Outparam[2]['weight_decay'] = weight_decay
    return Outparam

def create_module(module_str):
    tmpss = module_str.split(",")
    assert len(tmpss) == 2, "Error formate\
        of the module path: {}".format(module_str)
    module_name, function_name = tmpss[0], tmpss[1]
    somemodule = importlib.import_module(module_name, __package__)
    function = getattr(somemodule, function_name)
    return function

def tensor2img(imgGPU):
    imgCPU = imgGPU.cpu().transpose(0,1).transpose(1,2).numpy()
    imgCPU = (imgCPU*0.5+0.5)*255
    return imgCPU.astype(np.uint8)

def randomChoice(text,errorImg):
    try:
        text = np.array(text)
        errorImg = np.array(errorImg)
        index = np.random.choice(list(range(len(text))),min(5,len(text)),replace=False)
        text,errorImg = text[np.array(index)],errorImg[np.array(index)]
    except:
        return [],[]
    return text,errorImg

def loadkeyfile(keyfile):
    alphabet = ''.join([x.strip('\n') for x in list(open(keyfile, 'r', encoding='utf-8'))])
    alphabet = alphabet.replace('\ufeff', '').replace('\u3000', '').strip()
    return alphabet

def save_checkpoint(state,checkpoint, filename):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    
def makedir(path):
    if not os.path.exists(path):
        os.mkdir(path)

class averager(object):
    """Compute average for `torch.Variable` and `torch.Tensor`. """

    def __init__(self):
        self.reset()

    def add(self, v):
        if isinstance(v, Variable):
            count = v.data.numel()
            v = v.data.sum()
        elif isinstance(v, torch.Tensor):
            count = v.numel()
            v = v.sum()

        self.n_count += count
        self.sum += v

    def reset(self):
        self.n_count = 0
        self.sum = 0

    def val(self):
        res = 0
        if self.n_count != 0:
            res = self.sum / float(self.n_count)
        return res

