import torch
import math
import nltk
import torch.nn as nn
import sys

from utils import *
from tqdm import tqdm
from PIL import Image
from wp_utils import *
from attack import FGSM_REG
from timm.data import Mixup
from einops import rearrange
from typing import Iterable, Optional
from timm.utils import accuracy, AverageMeter
from nltk.translate.bleu_score import sentence_bleu
####################################
beta = 1.0

def get_loss_scale_for_deepspeed(model):
    optimizer = model.optimizer
    return optimizer.loss_scale if hasattr(optimizer, "loss_scale") else optimizer.cur_scale

@torch.no_grad()
def evaluate(net: torch.nn.Module, dataloader: Iterable, 
                  device: torch.device, criterion: torch.nn.Module, train_type='fim', if_attack=False, print_freq=10):
    net.eval()
    acc_meter = AverageMeter()
    loss_meter = AverageMeter()
    attack =FGSM_REG(net, 12./255., 2./255., min_val=0, max_val=1, max_iters=8)
    with torch.no_grad():
        for batch_idx, (imgs, targets) in enumerate(dataloader):
            imgs, bm_pos = imgs
            imgs, targets = imgs.to(device), targets.to(device)  
            bm_pos = bm_pos.to(device, non_blocking=True).flatten(1).to(torch.bool)
         
            if if_attack:
                bum_pos = torch.zeros_like(bm_pos)
                bum_pos = bum_pos.to(device, non_blocking=True).flatten(1).to(torch.bool)  
                per_data = attack.perturb(imgs, targets, bum_pos, 'mean', random_start=False, beta=beta)
                imgs = per_data
            outputs = net(img=imgs, bm_pos=bm_pos, target=targets, _eval=True) 
            outputs_x = outputs['out_x']
            loss = criterion(outputs_x, targets)
            batch_size = targets.size(0)

            idx, predicted = outputs_x.max(1)
            acc_meter.update(predicted.eq(targets).float().mean().item(), n=batch_size)
            loss_meter.update(loss.item(), 1)
            if batch_idx % print_freq == 0:
                print('Test %d/%d: [loss: %.4f] [acc1: %.3f/100]' %(batch_idx*batch_size, 
                        len(dataloader.dataset), loss_meter.avg, acc_meter.avg*100))   
    test_stat = {'loss': loss_meter.avg,
        'acc': acc_meter.avg}  
    return test_stat
    

def train_class_batch(model, samples, targets, bm_pos, criterion, train_type):
    if train_type.startswith('std'):
        outputs = model(img=samples, bm_pos=bm_pos, _eval=False)
        outputs_x = outputs['out_x']
        loss = criterion(outputs_x, targets)
    elif train_type.startswith('fim'):
        outputs = model(img=samples, bm_pos=bm_pos, target=targets, _eval=False)
        outputs_x = outputs['out_x']
        if 'out_c' in outputs.keys():
            fim_loss = 0.
            for extra_output in outputs['out_c']:
                fim_loss += F.cross_entropy(extra_output, targets)
            fim_loss = fim_loss / len(outputs['out_c'])
            loss = criterion(outputs_x, targets)
            loss += beta * fim_loss
        if 'vq_loss' in outputs.keys():
            loss += outputs['vq_loss']
    return loss, outputs_x


def train_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                data_loader: Iterable, optimizer: torch.optim.Optimizer,
                device: torch.device, epoch: int, loss_scaler, train_type, if_attack, max_norm: float=0,
                start_steps=None,lr_schedule_values=None, wd_schedule_values=None, 
                update_freq=None, print_freq=50):
    model.train(True)                                                         
    acc_meter = AverageMeter()
    loss_meter = AverageMeter()
    
    attack = FGSM_REG(model, 8./255., 2./255., min_val=0, max_val=1, max_iters=4)

    if loss_scaler is None:    
        model.zero_grad()
        model.micro_steps = 0
    else:
        optimizer.zero_grad()

    for data_iter_step, (samples ,targets) in enumerate(data_loader):    
        step = data_iter_step // update_freq
        it = start_steps + step  
        if lr_schedule_values is not None or wd_schedule_values is not None and data_iter_step % update_freq == 0:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]                
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]
        samples, bm_pos = samples
        targets = targets.to(device, non_blocking=True)
        samples = samples.to(device, non_blocking=True)
        
        bm_pos = bm_pos.to(device, non_blocking=True).flatten(1).to(torch.bool)
        
        if if_attack:
            bum_pos = torch.zeros_like(bm_pos)
            bum_pos = bum_pos.to(device, non_blocking=True).flatten(1).to(torch.bool)     
            per_data = attack.perturb(samples, targets, bum_pos, 'mean', random_start=True, beta=beta)
            samples = per_data
               
        batch_size = samples.size(0)
                                                   
        with torch.cuda.amp.autocast():
            loss, outputs = train_class_batch(
                model, samples, targets, bm_pos, criterion, train_type)
        loss_value = loss.item()

        ######  Error                              
        if not math.isfinite(loss_value):   
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)
        ######  Update
        if loss_scaler is None:
            loss /= update_freq
            model.backward(loss)
            model.step()
        else:
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss /= update_freq
            grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                    parameters=model.parameters(), create_graph=is_second_order,
                                    update_grad=(data_iter_step + 1) % update_freq == 0)
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.zero_grad()

        torch.cuda.synchronize()    

        min_lr,max_lr = 10., 0.
        for group in optimizer.param_groups:
            min_lr,max_lr = min(min_lr, group["lr"]),max(max_lr, group["lr"])

        acc_meter.update((outputs.max(-1)[-1] == targets).float().mean().item(), n=batch_size)
        loss_meter.update(loss_value, 1)
        
        if data_iter_step % print_freq == 0:
            print('Epoch:[%d] %d/%d: [loss: %.3f] [acc1: %.3f /100] [lr: %.3e]' 
                %(epoch, batch_size*data_iter_step, len(data_loader.dataset),
                    loss_meter.avg, acc_meter.avg*100, max_lr))
            
    train_stat = {'loss': loss_meter.avg,
        'acc': acc_meter.avg}

    return train_stat 

def train_epoch_wp(model: torch.nn.Module, criterion: torch.nn.Module,
                data_loader: Iterable, optimizer: torch.optim.Optimizer,
                device: torch.device, epoch: int, loss_scaler, train_type, if_attack, wp_adver, max_norm: float=0,
                start_steps=None,lr_schedule_values=None, wd_schedule_values=None, 
                update_freq=None, print_freq=50):
    model.train(True)                                                         
    acc_meter = AverageMeter()
    loss_meter = AverageMeter()
    
    attack = FGSM_REG(model, 8./255., 2./255., min_val=0, max_val=1, max_iters=4)

    if loss_scaler is None:    
        model.zero_grad()
        model.micro_steps = 0
    else:
        optimizer.zero_grad()

    
    
    for data_iter_step, (samples ,targets) in enumerate(data_loader):    
        step = data_iter_step // update_freq
        it = start_steps + step  
        if lr_schedule_values is not None or wd_schedule_values is not None and data_iter_step % update_freq == 0:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]                
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]
        samples, bm_pos = samples
        targets = targets.to(device, non_blocking=True)
        samples = samples.to(device, non_blocking=True)
        
        bm_pos = bm_pos.to(device, non_blocking=True).flatten(1).to(torch.bool)
        
        if if_attack:
            bum_pos = torch.zeros_like(bm_pos)
            bum_pos = bum_pos.to(device, non_blocking=True).flatten(1).to(torch.bool)     
            per_data = attack.perturb(samples, targets, bum_pos, 'mean', random_start=True, beta=beta)
            samples = per_data
        
        if epoch >= 5:
            awp = wp_adver.calc_awp(inputs_adv=samples,
                                            targets=targets)
            wp_adver.perturb(awp)
                                                                      
        batch_size = samples.size(0)
                                                   
        with torch.cuda.amp.autocast():
            loss, outputs = train_class_batch(
                model, samples, targets, bm_pos, criterion, train_type)
        loss_value = loss.item()

        ######  Error                              
        if not math.isfinite(loss_value):   
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)
        ######  Update
        if loss_scaler is None:
            loss /= update_freq
            model.backward(loss)
            model.step()
        else:
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss /= update_freq
            grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                    parameters=model.parameters(), create_graph=is_second_order,
                                    update_grad=(data_iter_step + 1) % update_freq == 0)
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.zero_grad()

        torch.cuda.synchronize()    

        min_lr,max_lr = 10., 0.
        for group in optimizer.param_groups:
            min_lr,max_lr = min(min_lr, group["lr"]),max(max_lr, group["lr"])

        if epoch >= 10:
            wp_adver.restore(awp)
                
        acc_meter.update((outputs.max(-1)[-1] == targets).float().mean().item(), n=batch_size)
        loss_meter.update(loss_value, 1)
        
        if data_iter_step % print_freq == 0:
            print('Epoch:[%d] %d/%d: [loss: %.3f] [acc1: %.3f /100] [lr: %.3e]' 
                %(epoch, batch_size*data_iter_step, len(data_loader.dataset),
                    loss_meter.avg, acc_meter.avg*100, max_lr))
            
    train_stat = {'loss': loss_meter.avg,
        'acc': acc_meter.avg}

    return train_stat 


