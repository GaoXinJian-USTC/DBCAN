# Copyright (c) OpenMMLab. All rights reserved.
"""This code is from https://github.com/jadore801120/attention-is-all-you-need-
pytorch."""
import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention module."""

    def __init__(self,
                 n_head=8,
                 d_model=512,
                 d_k=64,
                 d_v=64,
                 dropout=0.1,
                 qkv_bias=False,
                 mask_value=0):
        super().__init__()

        self.mask_value = mask_value

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.scale = d_k**-0.5

        self.dim_k = n_head * d_k
        self.dim_v = n_head * d_v

        self.linear_q = nn.Linear(self.dim_k, self.dim_k, bias=qkv_bias)

        self.linear_k = nn.Linear(self.dim_k, self.dim_k, bias=qkv_bias)

        self.linear_v = nn.Linear(self.dim_v, self.dim_v, bias=qkv_bias)

        self.fc = nn.Linear(self.dim_v, d_model, bias=qkv_bias)

        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        
        batch_size, len_q, _ = q.size()
        _, len_k, _ = k.size()

        q = self.linear_q(q).view(batch_size, len_q, self.n_head, self.d_k)
        k = self.linear_k(k).view(batch_size, len_k, self.n_head, self.d_k)
        v = self.linear_v(v).view(batch_size, len_k, self.n_head, self.d_v)

        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 3, 1)
        v = v.permute(0, 2, 1, 3)

        logits = torch.matmul(q, k) * self.scale

        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            elif mask.dim() == 2:
                mask = mask.unsqueeze(1).unsqueeze(1)
            logits = logits.masked_fill(mask == self.mask_value, float('-inf'))
        weights = logits.softmax(dim=-1)
        weights = self.attn_drop(weights)

        attn_out = torch.matmul(weights, v).transpose(1, 2)
        attn_out = attn_out.reshape(batch_size, len_q, self.dim_v)
        attn_out = self.fc(attn_out)
        attn_out = self.proj_drop(attn_out)

        return attn_out


class PositionwiseFeedForward(nn.Module):
    """A two-feed-forward-layer module."""

    def __init__(self, d_in, d_hid, dropout=0.1, act_layer=nn.GELU):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)
        self.w_2 = nn.Linear(d_hid, d_in)
        self.act = act_layer()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.w_1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.w_2(x)
        x = self.dropout(x)

        return x


def get_pad_mask(seq, pad_idx):
    return (seq != pad_idx).unsqueeze(-2)


def get_subsequent_mask(seq):
    """For masking out the subsequent info."""
    len_s = seq.size(1)
    subsequent_mask = 1 - torch.triu(
        torch.ones((len_s, len_s), device=seq.device), diagonal=1)
    subsequent_mask = subsequent_mask.unsqueeze(0).bool()
    return subsequent_mask


class CrossAttentionFusionLayer(nn.Module):

    def __init__(self,
                 d_model=512,
                 d_inner=256,
                 n_head=8,
                 d_k=64,
                 d_v=64,
                 dropout=0.1,
                 qkv_bias=False,
                 mask_value=0,
                 act_layer=nn.GELU):
        super().__init__()
        
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.enc_attn = MultiHeadAttention(
            n_head,
            d_model,
            d_k,
            d_v,
            dropout=dropout,
            qkv_bias=qkv_bias,
            mask_value=mask_value)
        self.mlp = PositionwiseFeedForward(
            d_model, d_inner, dropout=dropout, act_layer=act_layer)

    def forward(self,
                dec_input,
                enc_output,
                self_attn_mask=None,
                dec_enc_attn_mask=None):

        enc_attn_q = self.norm2(dec_input)
        enc_attn_out = self.enc_attn(enc_attn_q, enc_output, enc_output,
                                     dec_enc_attn_mask)

        mlp_in = dec_input + enc_attn_out
        mlp_out = self.mlp(self.norm3(mlp_in))
        out = mlp_in + mlp_out

        return out


