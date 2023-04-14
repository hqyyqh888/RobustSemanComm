import sys

sys.path.append("..")

import os
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

from utils import tensor2cuda


def project(x, original_x, epsilon, _type='linf'):
    if _type == 'linf':
        max_x = original_x + epsilon
        min_x = original_x - epsilon
        x = torch.max(torch.min(x, max_x), min_x)

    elif _type == 'l2':
        dist = (x - original_x)
        dist = dist.view(x.shape[0], -1)
        dist_norm = torch.norm(dist, dim=1, keepdim=True)
        mask = (dist_norm > epsilon).unsqueeze(2).unsqueeze(3)
        # dist = F.normalize(dist, p=2, dim=1)
        dist = dist / dist_norm
        dist *= epsilon
        dist = dist.view(x.shape)
        x = (original_x + dist) * mask.float() + x * (1 - mask.float())
    else:
        raise NotImplementedError
    return x


class FastGradientSignUntargeted():
    """
        Fast gradient sign untargeted adversarial attack, minimizes the initial class activation
        with iterative grad sign updates
    """

    def __init__(self, model, epsilon, alpha, min_val, max_val, max_iters, _type='linf'):
        self.model = model
        # self.model.eval()

        # Maximum perturbation
        self.epsilon = epsilon
        # Movement multiplier per iteration
        self.alpha = alpha
        # Minimum value of the pixels
        self.min_val = min_val
        # Maximum value of the pixels
        self.max_val = max_val
        # Maximum numbers of iteration to generated adversaries
        self.max_iters = max_iters
        # The perturbation of epsilon
        self._type = _type


    def perturb(self, original_images, labels, bm_pos, reduction4loss='mean', random_start=False, beta=2):
        # original_images: values are within self.min_val and self.max_val
        # The adversaries created from random close points to the original data
        if random_start:
            rand_perturb = torch.FloatTensor(original_images.shape).uniform_(
                -self.epsilon, self.epsilon)
            rand_perturb = tensor2cuda(rand_perturb)
            x = original_images + rand_perturb
            x.clamp_(self.min_val, self.max_val)
        else:
            x = original_images.clone()
        x.requires_grad = True
        self.model.eval()

        with torch.enable_grad():
            for _iter in range(self.max_iters):
                outputs = self.model(x, bm_pos, labels, _eval=True)
                outputs_x= outputs['out_x']
                loss = F.cross_entropy(outputs_x, labels, reduction=reduction4loss)
                fim_loss = 0.
                if 'out_c' in outputs.keys():
                    for extra_output in outputs['out_c']:
                        fim_loss += F.cross_entropy(extra_output, labels, reduction=reduction4loss)
                    if len(outputs['out_c']) > 0:
                        fim_loss /= len(outputs['out_c'])
                        
                    loss += beta * fim_loss
                if 'vq_loss' in outputs.keys():
                    loss += outputs['vq_loss']

                if reduction4loss == 'none':
                    grad_outputs = tensor2cuda(torch.ones(loss.shape))

                else:
                    grad_outputs = None

                grads = torch.autograd.grad(loss, x, grad_outputs=grad_outputs,
                                            only_inputs=True)[0]
                x.data += self.alpha * torch.sign(grads.data)
                # the adversaries' pixel value should within max_x and min_x due
                # to the l_infinity / l2 restriction
                x = project(x, original_images, self.epsilon, self._type)
                x.clamp_(self.min_val, self.max_val)
        self.model.train()
        return x


def norm_psr(v, Rx, psr):
    x_square = torch.mul(Rx, Rx)
    power1 = torch.mean(x_square)
    x_square = torch.mul(v, v)
    power2 = torch.mean(x_square)
    psr = 10 ** (-psr / 10)
    v = v * (power1 * psr / power2).sqrt()
    return v


def FGSM_symbol_attack(pert, model, images, labels, mask_pos, psr):
    x = images.clone()
    reduction4loss = 'mean'
    alpha = 0.5
    x.requires_grad = True
    model.eval()
    with torch.enable_grad():
        for _iter in range(5):
            outputs,_, Rxsig = model(x, mask_pos, pert, psr)
            # outputs, class_wise_output = self.model(x, _eval=True)
            loss = F.cross_entropy(outputs, labels, reduction=reduction4loss)
            if reduction4loss == 'none':
                grad_outputs = tensor2cuda(torch.ones(loss.shape))
            else:
                grad_outputs = None
            grads = torch.autograd.grad(loss, Rxsig, grad_outputs=grad_outputs,
                                        only_inputs=True)[0]
            pert += alpha * torch.sign(grads.data)
            pert = norm_psr(pert, Rxsig, psr)
        return pert




def FGSM_attack_codebook(pert, model, images, labels, mask_pos, psr):
    x = images.clone()
    reduction4loss = 'mean'
    x.requires_grad = True
    model.eval()
    wmbweight = model.vq_layer.embedding.weight
    with torch.enable_grad():
        for _iter in range(1):
            outputs,class_wise_output,vq_loss,_, Rx_index = model(x, mask_pos, labels, pert, attackmimo = False, _eval=True)
            # outputs, class_wise_output = self.model(x, _eval=True)

            loss = F.cross_entropy(outputs, labels, reduction=reduction4loss)
            #loss+= vq_loss
            channel_reg_loss = 0.
            for extra_output in class_wise_output:
                channel_reg_loss += F.cross_entropy(extra_output, labels, reduction=reduction4loss)
                # channel_reg_loss += F.cross_entropy(extra_output, reduction=reduction4loss)
            if len(class_wise_output) > 0:
                channel_reg_loss /= len(class_wise_output)
            #print(vq_loss)
           # loss += 1 * channel_reg_loss
            if reduction4loss == 'none':
                grad_outputs = tensor2cuda(torch.ones(loss.shape))

            else:
                grad_outputs = None
            grads = torch.autograd.grad(loss, Rx_index, grad_outputs=grad_outputs,
                                        only_inputs=True)[0]
            index = torch.zeros((196,1), dtype = torch.int64)
            for i in range(Rx_index.shape[1]):
                temp = Rx_index[0,i,:] - wmbweight
                find_index = torch.mm(temp, grads.data[0, i].unsqueeze(0).t())/(torch.norm(temp, dim=1, keepdim=True)*torch.norm(grads.data[0, i]))
                find_index = torch.max(find_index, dim=0)[1]
                index[i,0] = find_index
            pert = index
        return pert        
