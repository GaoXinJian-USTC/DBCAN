# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F


class DotProductAttentionLayer(nn.Module):

    def __init__(self, dim_model=None):
        super().__init__()

        self.scale = dim_model**-0.5 if dim_model is not None else 1.

    def forward(self, query, key, value, mask=None):
        n, seq_len = mask.size()
        # print(f"{query.permute(0,2,1).shape} * {key.shape}")
        logits = torch.matmul(query.permute(0, 2, 1), key) * self.scale
        # print(f"{query.permute(0,2,1).shape} * {key.shape} = {logits.shape}")

        if mask is not None:
            mask = mask.view(n, 1, seq_len)
            # print(mask.shape)
            logits = logits.masked_fill(mask, float('-inf'))

        weights = F.softmax(logits, dim=2)
        
        glimpse = torch.matmul(weights, value.transpose(1, 2))
        # print(f"{weights.shape} * {value.shape} = {glimpse.shape}")
        glimpse = glimpse.permute(0, 2, 1).contiguous() #24 30 128
        return glimpse
