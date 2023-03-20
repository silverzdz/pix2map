from collections import OrderedDict
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

class LayerNorm(nn.LayerNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)
    
class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(1.702 * x)
    
class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, key_padding_mask: torch.Tensor = None) -> None:
        super().__init__()
        
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask
        self.key_padding_mask = key_padding_mask
        
    def attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        self.attn_mask = self.attn_mask.to(dtype=k.dtype, device=k.device) if self.attn_mask is not None else None
        return self.attn(q, k, v, key_padding_mask = self.key_padding_mask, need_weights=False, attn_mask=self.attn_mask)[0]
    
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        ###TODO: true??
        ### important!!!
        ### k,q,v's shape should be [512] but not [1, 512]!!! 
        #assert k.shape[0] == 512 and q.shape[0] == 512 and v.shape[0] == 512
        
        ### attn_mask: [N,num_heads,L,S]
        # N: batch_size
        # L: target sequence length
        # S: source sequence length
        x = q + self.attention(self.ln_1(q), self.ln_1(k), self.ln_1(v))
        x = x + self.mlp(self.ln_2(x))
        return x, k, x
    
class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None, key_padding_mask: torch.Tensor = None) -> None:
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblock = ResidualAttentionBlock(width, heads, attn_mask, key_padding_mask)

    def forward(self, v_0: torch.Tensor) -> torch.Tensor:
        assert v_0.shape[2] == 512
        v_l = v_0.permute(1,0,2)
        for l in range(self.layers):
            v_l = self.resblock(v_l, v_l, v_l)[0]
        v_l = v_l.permute(1,0,2)
        v = torch.mean(v_l, dim = 1)
        return v