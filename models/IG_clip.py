from collections import OrderedDict
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .resnet import ModifiedResNet
from .transformer import Transformer, LayerNorm


class ImageGraphClip(nn.Module):
    
    def __init__(self,
                 embed_dim: int,
                 image_resolution: int,
                 resnet_layers: Tuple[int, int, int, int],
                 img_num: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int,
                 attn_mask: torch.Tensor = None
                 ) -> None:
        super().__init__()
        
        self.embed_dim = embed_dim
        self.image_resolution = image_resolution
        self.resnet_layers = resnet_layers
        self.img_num = img_num
        self.transformer_width = transformer_width
        self.transformer_heads = transformer_heads
        self.transformer_layers = transformer_layers
        self.attn_mask = attn_mask
        
        self.token_embedding = torch.nn.Embedding(10000, embed_dim, padding_idx=0)
        self.ln_final = LayerNorm(transformer_width)
        self.cos = nn.CosineSimilarity(dim = 0)
        
        self.resnet = ModifiedResNet(resnet_layers, embed_dim, transformer_heads, img_num, image_resolution)
        self.transformer = Transformer(transformer_width, transformer_layers, transformer_heads, attn_mask)
        
        self.init_parameters()
        
    def init_parameters(self) -> None:
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        
        if self.resnet.attnpool is not None:
            std = self.resnet.attnpool.c_proj.in_features ** -0.5
            nn.init.normal_(self.resnet.attnpool.q_proj.weight, std=std)
            nn.init.normal_(self.resnet.attnpool.k_proj.weight, std=std)
            nn.init.normal_(self.resnet.attnpool.v_proj.weight, std=std)
            nn.init.normal_(self.resnet.attnpool.c_proj.weight, std=std)
            
        for resnet_block in [self.resnet.layer1, self.resnet.layer2, self.resnet.layer3, self.resnet.layer4]:
            for name, param in resnet_block.named_parameters():
                if name.endswith("bn3.weight"):
                    nn.init.zeros_(param)
                    
        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)
    
    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        return self.resnet(image)
    
    def encode_graph(self, graph: torch.Tensor, adj_matrix: torch.Tensor) -> torch.Tensor:
        x = self.token_embedding(graph)
        
        x = x.permute(1,0,2)
        x = self.transformer(x, adj_matrix)
        x = x.permute(1,0,2)
        x = torch.mean(x, dim = 1)
        x = self.ln_final(x)
        
        return x
    
    def forward(self, image: torch.Tensor, graph:torch.Tensor, adj_matrix: torch.Tensor) -> torch.Tensor:
        image_features = self.encode_image(image)
        graph_features = self.encode_graph(graph, adj_matrix)
        
        # image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        # graph_features = graph_features / graph_features.norm(dim=-1, keepdim=True)
        
        batch_size = image_features.shape[0]
        
        cos_matrix = torch.zeros((batch_size, batch_size)).to(image_features.device)
        for i in range(batch_size):
            for j in range(batch_size):
                cos_matrix[i][j] = self.cos(image_features[i], graph_features[j])
        
        return cos_matrix