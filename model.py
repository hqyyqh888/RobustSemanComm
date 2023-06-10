import math
import torch
import pickle
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from channel import *
from model_util import *
from functools import partial
from model_util import _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_ as __call_trunc_normal_
from typing import List, Callable, Union, Any, TypeVar, Tuple
from transformers import  BertModel

from base_args import IMGC_NUMCLASS


def trunc_normal_(tensor, mean=0., std=1.):
    __call_trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)

__all__ = [
    'ViT_Van_model',
    'ViT_FIM_model']

class ViT_Van_CLS(nn.Module):
    def __init__(self,
                 img_size=224, patch_size=16, encoder_in_chans=3, encoder_num_classes=0, 
                 encoder_embed_dim=768, encoder_depth=12,encoder_num_heads=12, decoder_num_classes=768, 
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=8, mlp_ratio=4., 
                 qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0., 
                 norm_layer=nn.LayerNorm, init_values=0.,use_learnable_pos_emb=False,num_classes=0, 
                 ):
        super().__init__()
        self.img_encoder = ViTEncoder_Van(img_size=img_size, patch_size=patch_size, in_chans=encoder_in_chans, 
                                num_classes=encoder_num_classes, embed_dim=encoder_embed_dim,depth=encoder_depth,
                                num_heads=encoder_num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,drop_rate=drop_rate, 
                                drop_path_rate=drop_path_rate,norm_layer=norm_layer, init_values=init_values,
                                use_learnable_pos_emb=use_learnable_pos_emb)
        
        self.img_decoder = ViTDecoder_Van(patch_size=patch_size, num_patches=self.img_encoder.patch_embed.num_patches,
                                num_classes=decoder_num_classes, embed_dim=decoder_embed_dim, depth=decoder_depth,
                                num_heads=decoder_num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, 
                                drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate, 
                                norm_layer=norm_layer,init_values=init_values) if decoder_depth>0 else nn.Identity()
        
        
        
        self.encoder_to_channel = nn.Linear(encoder_embed_dim, 32)
        self.channel = Channels()
        self.channel_to_decoder = nn.Linear(32, decoder_embed_dim)
        # self.head = nn.Linear(decoder_embed_dim, IMGC_NUMCLASS)
        self.head = nn.Linear(decoder_embed_dim, IMGC_NUMCLASS)
       
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'mask_token'}

    def forward(self, img, bm_pos, target=None, _eval=False, test_snr=200):
        if _eval:
            self.eval()
        else:
            self.train()
        out = {}
        if self.training:
            noise_snr, noise_var = noise_gen(self.training)
            noise_var,noise_snr = noise_var.cuda(), noise_snr.cpu().item()
        else:
            noise_var = torch.FloatTensor([1]) * 10**(-test_snr/20)  
        x = self.img_encoder(img, bm_pos)
        x = self.encoder_to_channel(x)
        # x = power_norm_batchwise(x)
        # x = self.channel.AWGN(x, noise_var.item())
        x = self.channel_to_decoder(x)
        x = self.img_decoder(x)
        # x = self.head(x.view(x.shape[0],-1))
        x = self.head(x.mean(1))
        out['out_x'] = x
        return out

class ViT_FIM_CLS(nn.Module):
    def __init__(self,
                 img_size=224, patch_size=16, encoder_in_chans=3, encoder_num_classes=0, 
                 encoder_embed_dim=768, encoder_depth=12,encoder_num_heads=12, decoder_num_classes=768, 
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=8, mlp_ratio=4., 
                 qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0., 
                 norm_layer=nn.LayerNorm, init_values=0.,use_learnable_pos_emb=False,num_classes=0, 
                 ):
        super().__init__()
        self.img_encoder = ViTEncoder_FIM(img_size=img_size, patch_size=patch_size, in_chans=encoder_in_chans, 
                                num_classes=encoder_num_classes, embed_dim=encoder_embed_dim,depth=encoder_depth,
                                num_heads=encoder_num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,drop_rate=drop_rate, 
                                drop_path_rate=drop_path_rate,norm_layer=norm_layer, init_values=init_values,
                                use_learnable_pos_emb=use_learnable_pos_emb)
        self.img_decoder = ViTDecoder_Van(patch_size=patch_size, num_patches=self.img_encoder.patch_embed.num_patches,
                                num_classes=decoder_num_classes, embed_dim=decoder_embed_dim, depth=decoder_depth,
                                num_heads=decoder_num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, 
                                drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate, 
                                norm_layer=norm_layer,init_values=init_values) if decoder_depth>0 else nn.Identity()
        
        self.encoder_to_channel = nn.Linear(encoder_embed_dim, 32)
        self.channel = Channels()
        self.channel_to_decoder = nn.Linear(32, decoder_embed_dim)
        self.head = nn.Linear(decoder_embed_dim*196, IMGC_NUMCLASS)
        self.bit_per_digit = 12
        self.vq_layer = VectorQuantizer(num_embeddings=2**self.bit_per_digit,
                                        embedding_dim=32)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'mask_token'}

    def forward(self, img, bm_pos, target=None, _eval=False, test_snr=200):
        if _eval:
            self.eval()
        else:
            self.train()
        out = {}
        
        if self.training:
            noise_snr, noise_var = noise_gen(self.training)
            noise_var,noise_snr = noise_var.cuda(), noise_snr.cpu().item()
        else:
            noise_var = torch.FloatTensor([1]) * 10**(-test_snr/20)  
            noise_snr = test_snr
        x, cls_out = self.img_encoder(img, bm_pos, target)
        x = self.encoder_to_channel(x)
        # x, vq_loss = self.vq_layer(x, noise_snr, self.bit_per_digit)
        # x = power_norm_batchwise(x)
        # x = self.channel.AWGN(x, noise_var.item())
        x = self.channel_to_decoder(x)
        x = self.img_decoder(x)
        x = self.head(x.view(x.shape[0],-1))
        # x = self.head(x.mean(1))
        out['out_x'] = x      
        out['out_c'] = cls_out
        # out['vq_loss'] = vq_loss
        return out
       

@register_model
def ViT_Van_model(pretrained=False, **kwargs):
    model = ViT_Van_CLS(
        img_size=224,
        patch_size=16,
        encoder_embed_dim=384,
        encoder_depth=4,
        encoder_num_heads=6,
        decoder_embed_dim=12,
        decoder_depth=1,
        decoder_num_heads=4,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model
        
@register_model
def ViT_FIM_model_S(pretrained=False, **kwargs):
    model = ViT_FIM_CLS(
        img_size=224,
        patch_size=16,
        encoder_embed_dim=384,
        encoder_depth=4,
        encoder_num_heads=6,
        decoder_embed_dim=12,
        decoder_depth=1,
        decoder_num_heads=4,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def ViT_FIM_model_L(pretrained=False, **kwargs):
    model = ViT_FIM_CLS(
        img_size=224,
        patch_size=16,
        encoder_embed_dim=784,
        encoder_depth=8,
        encoder_num_heads=6,
        decoder_embed_dim=12,
        decoder_depth=4,
        decoder_num_heads=8,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained: 
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model
