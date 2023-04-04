from collections import OrderedDict
from typing import List, Tuple, Dict
import math
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

def nodemap_to_array(map: Dict[Tuple[int, int], int]) -> np.ndarray:
    nodes = list(map.keys())
    return np.array([np.array(node) for node in nodes])

def cal_min_dist(v: torch.Tensor, pts: torch.Tensor, n_pts: int) -> Tuple[float, int]:
    
    #min_dist = np.min(np.array([np.linalg.norm(v-pts[i]) for i in range(pts.shape[0])]))
    
    min_dist = 1e20
    min_index = -1
    for i in range(n_pts):
        # dist = math.sqrt(math.pow(float(v[0])-float(pts[i][0]),2) + math.pow(float(v[1])-float(pts[i][1]),2))
        dist = math.sqrt(math.pow(v[0]-pts[i][0],2) + math.pow(v[1]-pts[i][1],2))
        if dist < min_dist:
            min_dist = dist
            min_index = i
    
    return float(min_dist), min_index

def contrastive_loss(cos_matrix: torch.Tensor) -> torch.Tensor:
    loss = torch.zeros(())
    n = cos_matrix.shape[0]
    for i in range(n):
        a_ii = cos_matrix[i][i]
        l_ig = torch.zeros(())
        l_gi = torch.zeros(())
        for j in range(n):
            l_ig = l_ig + torch.exp(cos_matrix[i][j])
            l_gi = l_gi + torch.exp(cos_matrix[j][i])
        l_ig = -torch.log((torch.exp(a_ii))/l_ig)
        l_gi = -torch.log((torch.exp(a_ii))/l_gi)
        loss = loss + l_ig + l_gi
    loss = loss / (n * 2.0)
    return loss

def chamfer_loss(cos_matrix: torch.Tensor, node_map: torch.Tensor, map_size: torch.Tensor) -> torch.Tensor:
    
    loss = torch.zeros(())
    n = cos_matrix.shape[0]
    # nodes_list = [nodemap_to_array(node_map) for node_map in node_maps]
    for i in range(n):
        nodes = node_map[i]
        sum_a_i = torch.zeros(())
        a_i_list = []
        for j in range(n):
            sum_a_i += torch.exp(cos_matrix[j][i])
        for j in range(n):
            a_i_list.append(torch.exp(cos_matrix[j][i])/sum_a_i)
        inner_loss = torch.zeros(())
        
        for k in range(int(map_size[i])):
            inner_inner_loss = torch.zeros(())
            for j in range(n):
                if j == i:
                    continue
                min_dist, _ = cal_min_dist(nodes[k], node_map[j], int(map_size[j]))
                inner_inner_loss += a_i_list[j] * min_dist
            inner_loss += inner_inner_loss/n
            
        loss += inner_loss/nodes.shape[0]

    loss = loss/n
    return loss
            
            
            
def edge_loss(cos_matrix: torch.Tensor, node_map: torch.Tensor, adj_matrixes: torch.Tensor, map_size: torch.Tensor) -> torch.Tensor:
    bce_loss = nn.BCELoss()
    loss = torch.zeros(())
    epsilon = 1e-8
    n = cos_matrix.shape[0]
    # nodes_list = [nodemap_to_array(node_map) for node_map in node_maps]
    
    for i in range(n):
        nodes = node_map[i]
        sum_a_i = torch.zeros(())
        a_i_list = []
        for j in range(n):
            sum_a_i += torch.exp(cos_matrix[j][i])
        for j in range(n):
            a_i_list.append(torch.exp(cos_matrix[j][i])/sum_a_i)
        inner_loss = torch.zeros(())
        
        cnt = 0
        for v in range(int(map_size[i])):
            ws = np.where(np.array(adj_matrixes[i][v].cpu()) == 0)[0].tolist()
            for w in ws:
                # inner_inner_loss = torch.zeros(())
                if v == w:
                    continue
                cnt += 1
                rnd = torch.ones(())
                    
                lnd = torch.zeros(())
                for j in range(n):
                    if j == i:
                        continue
                    _, v_index = cal_min_dist(nodes[v], node_map[j], map_size[j])
                    _, w_index = cal_min_dist(nodes[w], node_map[j], map_size[j])
                    if adj_matrixes[j][v_index][w_index] == 0:
                        lnd += a_i_list[j]
                lnd += epsilon
                inner_loss += bce_loss(lnd, rnd)
        
        inner_loss = inner_loss / cnt if cnt != 0 else torch.zeros(())
        loss += inner_loss
    
    loss = loss/n
    return loss