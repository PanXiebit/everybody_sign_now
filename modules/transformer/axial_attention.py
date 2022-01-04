


import math
import argparse
import numpy as np

import torch
import torch.nn as nn


from modules.transformer.attention import MultiHeadAttention
from modules.utils import shift_dim

class AxialBlock(nn.Module):
    def __init__(self, n_hiddens, n_head):
        super().__init__()
        kwargs = dict(shape=(0,) * 3, dim_q=n_hiddens,
                      dim_kv=n_hiddens, n_head=n_head,
                      n_layer=1, causal=False, attn_type='axial')
        self.attn_w = MultiHeadAttention(attn_kwargs=dict(axial_dim=-2),
                                         **kwargs)
        self.attn_h = MultiHeadAttention(attn_kwargs=dict(axial_dim=-3),
                                         **kwargs)
        self.attn_t = MultiHeadAttention(attn_kwargs=dict(axial_dim=-4),
                                         **kwargs)

    def forward(self, x): 
        """x : [bs, lengthn height, width, hid_dim]
        """
        # x = shift_dim(x, 1, -1)
        x = self.attn_w(x, x, x) + self.attn_h(x, x, x) + self.attn_t(x, x, x)
        # x = shift_dim(x, -1, 1)
        return x

if __name__ == "__main__":
    x = torch.randn(2, 4, 32, 32, 256)
    m = AxialBlock(256, 8)
    out = m(x)
    print(out.shape)