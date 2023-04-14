import os
import torch
import numpy as np
import pandas as pd
import torch.utils.data as data

from data import *
from dataset_folder import ImageFolder
from timm.data import create_transform
from torchvision import datasets, transforms
from torch.utils.data.sampler import RandomSampler


class RandomMaskingGenerator:
    def __init__(self, input_size, mask_ratio):
        if not isinstance(input_size, tuple):
            input_size = (input_size,) * 2
        self.height, self.width = input_size
        self.num_patches = self.height * self.width
        self.num_mask = int(mask_ratio * self.num_patches)

    def __repr__(self):
        repr_str = "Maks: total patches {}, mask patches {}".format(
            self.num_patches, self.num_mask
        )
        return repr_str

    def __call__(self):
        mask = np.hstack([
            np.zeros(self.num_patches - self.num_mask),
            np.ones(self.num_mask)])
        np.random.shuffle(mask)
        return mask 


class Gen_ma(object):
    def __init__(self, is_train, args ):
        self.transform = build_img_transform(is_train, args)
        self.masked_position_generator = RandomMaskingGenerator(
            args.window_size, args.mask_ratio)

    def __call__(self, image):
        return self.transform(image), self.masked_position_generator()

    def __repr__(self):
        repr += "  transform = %s,\n" % str(self.transform)
        repr += "  Masked position generator = %s,\n" % str(self.masked_position_generator)
        repr += ")"
        return repr


def build_dataset(is_train, args):
    transform = Gen_ma(is_train, args)
    if args.data_set.startswith('cifar_S32'):
        dataset = CIFAR_M(args.data_path, train=is_train, transform=transform, 
                                        download=True)
    elif args.data_set.startswith('cifar_S224'):
        root = 'data/CIFAR_S224/Train' if is_train else 'data/CIFAR_S224/Test'
        dataset = ImageFolder(root, transform=transform)
    elif args.data_set.startswith('imagenet'):
        root = 'data/Imagenet/Train' if is_train else 'data/Imagenet/test'
        dataset = ImageFolder(root, transform=transform)
    else:
        raise NotImplementedError()
    return dataset


def build_img_transform(is_train, args):
    if args.data_set.startswith('cifar_S32'):
        resize_im = args.input_size > 32
    elif args.data_set.startswith('cifar_S224'):
        resize_im = False
    mean = (0.,0.,0.)
    std =  (1.,1.,1.)

    t = []
    if resize_im:
        crop_pct = 1
        size = int(args.input_size / crop_pct)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)

