
import utils
import torch
import numpy as np
import torch.nn.functional as F

from PIL import Image
from tqdm import tqdm
from timm.data import Mixup
from typing import Iterable, Optional
from timm.utils import accuracy, ModelEma
from attack import FGSM_symbol_attack, FGSM_mimosymbol_attack

def get_loss_scale_for_deepspeed(model):
    optimizer = model.optimizer
    return optimizer.loss_scale if hasattr(optimizer, "loss_scale") else optimizer.cur_scale

def train_class_batch(model, samples, target, criterion, mask):
    outputs,vq_loss,_ = model(samples, mask)
    loss = criterion(outputs, target) + vq_loss
    return loss, outputs

def proj_lp(v, xi, p):
    if p == 2:
        v = v * min(1, xi/np.linalg.norm(v.flatten(1)))
    elif p == np.inf:
        v = np.sign(v) * np.minimum(abs(v), xi)
    else:
         raise ValueError('Values of p different from 2 and Inf are currently not supported...')
    return v

def norm_psr(v, Rx, psr):
    x_square = torch.mul(Rx, Rx)
    power1 = torch.mean(x_square)
    x_square = torch.mul(v, v)
    power2 = torch.mean(x_square)
    psr = 10 ** (-psr / 10)
    v = v * (power1 * psr / power2).sqrt()
    return v

@torch.no_grad()
def gen_per(train_loader, val_set, model, device):
    out_size = 20
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    model.eval()
    temp = torch.randn((1,out_size)).to(device, non_blocking=True)
    per_temp = torch.randn((1,out_size)).to(device, non_blocking=True)
    num = 0
    psr = -8
    best_acc = 100
    for batch in metric_logger.log_every(train_loader, 10, header):
        num+=1
        images = batch[0]
        target = batch[-1]
        target = target.to(device, non_blocking=True)
        images, bm_pos = images
        images = images.to(device, non_blocking=True)
        per_temp = torch.randn((1,out_size)).to(device, non_blocking=True)
        bum_pos = torch.zeros_like(bm_pos)
        bm_pos = bm_pos.to(device, non_blocking=True).flatten(1).to(torch.bool)          
        bum_pos = bum_pos.to(device, non_blocking=True).flatten(1).to(torch.bool)     
        per_temp = FGSM_symbol_attack(per_temp, model, images, target, bum_pos, psr)
        temp += per_temp
        temp = norm_psr(temp, model(images, bum_pos, temp, psr, retain=False)[2], psr)
 
        with torch.cuda.amp.autocast():
            output,_,Rxsig = model(images, bum_pos, temp, psr, retain=False)
            loss = criterion(output, target) 
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

        num_images = 10000
        batch_size = 1
        # Compute the estimated labels in batches
        if num %1000 == 0:
            ii = 0
            with torch.no_grad():
                acc = 0.
                for data_list, label in tqdm(val_set):
                    img_batch, bm_pos = data_list
                    label = label.to(device, non_blocking=True)
                    img_batch = img_batch.to(device, non_blocking=True)
                    bm_pos = bm_pos.to(device, non_blocking=True).flatten(1).to(torch.bool)   
                    bum_pos = torch.zeros_like(bm_pos)
                    bm_pos = bm_pos.to(device, non_blocking=True).flatten(1).to(torch.bool)          
                    bum_pos = bum_pos.to(device, non_blocking=True).flatten(1).to(torch.bool)
                    per_temp = torch.zeros((img_batch.shape[0],20)).to(device, non_blocking=True)
                    output = model(img_batch, bum_pos, temp, psr, retain=False)[0]
                    acc += accuracy(output, label, topk=(1, 5))[0] * batch_size
                # Compute the fooling rate
                acc = acc / num_images
                print('Accuracy = ', acc)
                if acc < best_acc:
                    best_acc = acc
                print('Best Fooling Acc = ', best_acc)
                pertbation_name = 'Test-{:.2f}-{:.2f}-{}.npy'.format(torch.sqrt((torch.mean(temp**2))).max(), acc*100, psr)
                np.save('response_PER/'+pertbation_name, temp.cpu().numpy())
                
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}



