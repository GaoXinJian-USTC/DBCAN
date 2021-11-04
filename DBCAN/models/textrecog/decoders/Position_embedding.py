import math

import torch
import torch.nn as nn
from DBCAN.models.builder import DECODERS
from mmcv.runner import BaseModule
import numpy as np

class Adaptive2DPositionalEncoding(BaseModule):

    def __init__(self,
                 d_hid=128,
                 n_height=100,
                 n_width=100,
                 dropout=0.1,
                 init_cfg=[dict(type='Xavier', layer='Conv2d')]):
        super().__init__(init_cfg=init_cfg)

        h_position_encoder = self._get_sinusoid_encoding_table(n_height, d_hid)
        
        h_position_encoder = h_position_encoder.transpose(0, 1)
        h_position_encoder = h_position_encoder.view(1, d_hid, n_height, 1)

        w_position_encoder = self._get_sinusoid_encoding_table(n_width, d_hid)
        w_position_encoder = w_position_encoder.transpose(0, 1)
        w_position_encoder = w_position_encoder.view(1, d_hid, 1, n_width)

        self.register_buffer('h_position_encoder', h_position_encoder)
        self.register_buffer('w_position_encoder', w_position_encoder)
        self.h_scale = self.scale_factor_generate(d_hid)
        self.w_scale = self.scale_factor_generate(d_hid)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=dropout)

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        """Sinusoid position encoding table."""
        denominator = torch.Tensor([
            1.0 / np.power(10000, 2 * (hid_j // 2) / d_hid)
            for hid_j in range(d_hid)
        ])
        denominator = denominator.view(1, -1)
        pos_tensor = torch.arange(n_position).unsqueeze(-1).float()
        sinusoid_table = pos_tensor * denominator
        sinusoid_table[:, 0::2] = torch.sin(sinusoid_table[:, 0::2])
        sinusoid_table[:, 1::2] = torch.cos(sinusoid_table[:, 1::2])
        
        return sinusoid_table

    def scale_factor_generate(self, d_hid):
        scale_factor = nn.Sequential(
            nn.Conv2d(d_hid, d_hid, kernel_size=1), nn.ReLU(inplace=True),
            nn.Conv2d(d_hid, d_hid, kernel_size=1), nn.Sigmoid())

        return scale_factor

    def forward(self, x):

        b, c, h, w = x.size()

        avg_pool = self.pool(x)
        self.h_scale(avg_pool)
        # print(self.h_scale(avg_pool).shape,self.h_position_encoder[:, :, :h, :].shape)
        h_pos_encoding = \
            self.h_scale(avg_pool) * self.h_position_encoder[:, :, :h, :]

        w_pos_encoding = \
            self.w_scale(avg_pool) * self.w_position_encoder[:, :, :, :w]
        if c == 128:
            out = h_pos_encoding + w_pos_encoding

            out = self.dropout(out)

            return out
        else:
            out = x + h_pos_encoding + w_pos_encoding

            out = self.dropout(out)

            return out

class PosCNN(BaseModule):
    def __init__(self, in_chans, embed_dim=128, s=1,init_cfg=[dict(type='Xavier', layer='Conv2d')]):
        super().__init__(init_cfg=init_cfg)
        self.proj = nn.Sequential(nn.Conv2d(in_chans, embed_dim, 3, s, 1, bias=True, groups=embed_dim), nn.ReLU(inplace=True))
        self.s = s

    def forward(self, x):
        B, C, H , W = x.shape
        cnn_feat = x
        if self.s == 1:
            x = self.proj(cnn_feat) + cnn_feat
        else:
            x = self.proj(cnn_feat)
        return x

    def no_weight_decay(self):
        return ['proj.%d.weight' % i for i in range(4)]


class Adaptive3x3(BaseModule):
    def __init__(self,
                 d_hid=128,
                 n_height=48,
                 n_width=160,
                 dropout=0.1,
                 init_cfg=[dict(type='Xavier', layer='Conv2d')]):
        super().__init__(init_cfg=init_cfg)

        h_position_encoder = self._get_sinusoid_encoding_table(n_height, d_hid)
        
        h_position_encoder = h_position_encoder.transpose(0, 1)
        h_position_encoder = h_position_encoder.view(1, d_hid, n_height, 1)

        w_position_encoder = self._get_sinusoid_encoding_table(n_width, d_hid)
        w_position_encoder = w_position_encoder.transpose(0, 1)
        w_position_encoder = w_position_encoder.view(1, d_hid, 1, n_width)

        self.register_buffer('h_position_encoder', h_position_encoder)
        self.register_buffer('w_position_encoder', w_position_encoder)
        self.h_proj = nn.Sequential(nn.Conv2d(d_hid, d_hid, 3, 1, 1, bias=True, groups=d_hid), )
        self.w_proj = nn.Sequential(nn.Conv2d(d_hid, d_hid, 3, 1, 1, bias=True, groups=d_hid), )
        # self.h_proj = nn.Sequential(nn.Conv2d(d_hid, d_hid, 3, 1, 1, bias=True, groups=d_hid), nn.ReLU(inplace=True))
        # self.w_proj = nn.Sequential(nn.Conv2d(d_hid, d_hid, 3, 1, 1, bias=True, groups=d_hid), nn.ReLU(inplace=True))
        self.dropout = nn.Dropout(dropout)
    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        """Sinusoid position encoding table."""
        denominator = torch.Tensor([
            1.0 / np.power(10000, 2 * (hid_j // 2) / d_hid)
            for hid_j in range(d_hid)
        ])
        denominator = denominator.view(1, -1)
        pos_tensor = torch.arange(n_position).unsqueeze(-1).float()
        sinusoid_table = pos_tensor * denominator
        sinusoid_table[:, 0::2] = torch.sin(sinusoid_table[:, 0::2])
        sinusoid_table[:, 1::2] = torch.cos(sinusoid_table[:, 1::2])

        return sinusoid_table

    def forward(self, x):

        b, c, h, w = x.size()

        
        # print(self.h_scale(avg_pool).shape,self.h_position_encoder[:, :, :h, :].shape)
        h_pos_encoding = \
            self.h_proj(x) * self.h_position_encoder[:, :, :h, :]

        w_pos_encoding = \
            self.w_proj(x) * self.w_position_encoder[:, :, :, :w]
        
        out = h_pos_encoding + w_pos_encoding

        out = self.dropout(out)

        return out
       

