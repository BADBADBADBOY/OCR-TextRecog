# -*- coding: utf-8 -*-

import torch
import math

def AdamDecay(config, parameters):
    optimizer = torch.optim.Adam(parameters, lr=config['optimizer']['base_lr'],
                                 betas=(config['optimizer']['beta1'], config['optimizer']['beta2']),
                                 weight_decay=config['optimizer']['weight_decay'])
    return optimizer


def SGDDecay(config, parameters):
    optimizer = torch.optim.SGD(parameters, lr=config['optimizer']['base_lr'],
                                momentum=config['optimizer']['momentum'],
                                weight_decay=config['optimizer']['weight_decay'])
    return optimizer


def RMSPropDecay(config, parameters):
    optimizer = torch.optim.RMSprop(parameters, lr=config['optimizer']['base_lr'],
                                    alpha=config['optimizer']['alpha'],
                                    weight_decay=config['optimizer']['weight_decay'],
                                    momentum=config['optimizer']['momentum'])
    return optimizer

def AdamWDecay(config,parameters):
    optimizer = torch.optim.AdamW(parameters,lr=config['optimizer']['base_lr'],
                              betas=(config['optimizer']['beta1'], config['optimizer']['beta2']),
                              eps=config['optimizer']['eps'],
                              weight_decay=config['optimizer']['weight_decay'],
                              amsgrad=config['optimizer']['amsgrad'])
    return optimizer

def reset_base_lr(config,optimizer):
    optimizer.param_groups[0]['lr'] = config['optimizer']['base_lr']

def lr_poly(base_lr, epoch, max_epoch=1200, factor=0.9):
    return base_lr * ((1 - float(epoch) / max_epoch) ** (factor))

def lr_warm(base_lr, epoch, warm_epoch):
    return (base_lr/warm_epoch)*(epoch+1)

def adjust_learning_rate_poly(config, optimizer, epoch):
    lr = lr_poly(config['optimizer']['base_lr'], epoch - config['train']['warmepochs'],config['train']['epochs'] - config['train']['warmepochs'])
    optimizer.param_groups[0]['lr'] = lr
    if config['model']['STN']['STN_ON']:
        stn_lr = config['model']['STN']['stn_lr']
        lr = lr_poly(stn_lr, epoch - config['train']['warmepochs'],config['train']['epochs'] - config['train']['warmepochs'])
        optimizer.param_groups[1]['lr'] = lr

def adjust_learning_rate_warm(config, optimizer, epoch):
    lr = lr_warm(config['optimizer']['base_lr'], epoch,config['train']['warmepochs'])
    optimizer.param_groups[0]['lr'] = lr
    if config['model']['STN']['STN_ON']:
        stn_lr = config['model']['STN']['stn_lr']
        lr = lr_warm(stn_lr, epoch,config['train']['warmepochs'])
        optimizer.param_groups[1]['lr'] = lr

def adjust_learning_rate_step(config, optimizer, epoch):
    if epoch in config['optimizer']['schedule']:
        adjust_lr = optimizer.param_groups[0]['lr'] * config['optimizer']['gama']
        optimizer.param_groups[0]['lr'] = adjust_lr
        if config['model']['STN']['STN_ON']:
            stn_lr = config['model']['STN']['stn_lr']* config['optimizer']['gama']
            optimizer.param_groups[1]['lr'] = stn_lr

def adjust_learning_rate_cos(config, optimizer, epoch):
    initial_learning_rate,step,decay_steps,alpha = config['optimizer']['base_lr'],epoch - config['train']['warmepochs'],config['train']['epochs'] - config['train']['warmepochs'],config['optimizer']['alpha']
    step = min(step, decay_steps)
    cosine_decay = 0.5 * (1 + math.cos(math.pi * step / decay_steps))
    decayed = (1 - alpha) * cosine_decay + alpha
    optimizer.param_groups[0]['lr'] = initial_learning_rate * decayed
    
    if config['model']['STN']['STN_ON']:
        stn_lr = config['model']['STN']['stn_lr']* decayed
        optimizer.param_groups[1]['lr'] = stn_lr
     