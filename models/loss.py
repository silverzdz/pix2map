from collections import OrderedDict
from typing import List, Tuple, Dict
import math
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from ChamferDistancePytorch.chamfer2D import dist_chamfer_2D

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


### TODO: how to accelerate
def chamfer_loss_simple(cos_matrix: torch.Tensor, node_map: torch.Tensor, map_size: torch.Tensor) -> torch.Tensor:
    cham_loss = dist_chamfer_2D.chamfer_2DDist()
    
    loss = torch.zeros(()).to(cos_matrix.device)
    n = cos_matrix.shape[0]
    
    a = torch.zeros((n, n)).to(cos_matrix.device)
    for i in range(n):
        sum_a_i = torch.zeros(()).to(cos_matrix.device)
        for j in range(n):
            sum_a_i += torch.exp(cos_matrix[j][i])
        for j in range(n):
            a[i][j] = (torch.exp(cos_matrix[j][i])/sum_a_i)
    
    for i in range(n):
        nodes_1 = node_map[i:i+1][:,:map_size[i],:]
        
        for j in range(i, n):
            if j == i:
                continue
            nodes_2 = node_map[j:j+1][:,:map_size[j],:]
            dist1, dist2, idx1, idx2 = cham_loss(nodes_1, nodes_2)
            loss += a[i][j] * torch.mean(torch.sqrt(dist1)) + a[j][i] * torch.mean(torch.sqrt(dist2))
            
    loss = loss/(n*(n-1))
    
    return loss
            
            
            
def edge_loss(cos_matrix: torch.Tensor, node_map: torch.Tensor, adj_matrixes: torch.Tensor, map_size: torch.Tensor) -> torch.Tensor:
    bce_loss = nn.BCELoss()
    loss = torch.zeros(()).to(cos_matrix.device)
    epsilon = 1e-8
    n = cos_matrix.shape[0]
    # nodes_list = [nodemap_to_array(node_map) for node_map in node_maps]
    
    for i in range(n):
        nodes = node_map[i]
        sum_a_i = torch.zeros(()).to(cos_matrix.device)
        a_i_list = []
        for j in range(n):
            sum_a_i += torch.exp(cos_matrix[j][i])
        for j in range(n):
            a_i_list.append(torch.exp(cos_matrix[j][i])/sum_a_i)
        inner_loss = torch.zeros(()).to(cos_matrix.device)
        
        cnt = 0
        start = time.time()
        for v in range(int(map_size[i])):
            ws = np.where(np.array(adj_matrixes[i][v].cpu()) == 0)[0].tolist()
            for w in ws:
                # inner_inner_loss = torch.zeros(())
                if v == w:
                    continue
                cnt += 1
                rnd = torch.ones(()).to(cos_matrix.device)
                    
                lnd = torch.zeros(()).to(cos_matrix.device)
                for j in range(n):
                    if j == i:
                        continue
                    _, v_index = cal_min_dist(nodes[v], node_map[j], map_size[j])
                    _, w_index = cal_min_dist(nodes[w], node_map[j], map_size[j])
                    if adj_matrixes[j][v_index][w_index] == 0:
                        lnd += a_i_list[j]
                lnd += epsilon
                inner_loss += bce_loss(lnd, rnd)
        
        print("inner_loss: {}".format(inner_loss))
        end = time.time()
        print("Loss3 unit time: {}".format(end - start))
            
        inner_loss = inner_loss / cnt if cnt != 0 else torch.zeros(())
        loss += inner_loss
    
    loss = loss/n
    return loss

### TODO: calculate chamfer matrix in advance to accelerate
def edge_loss_simple(cos_matrix: torch.Tensor, node_map: torch.Tensor, adj_matrixes: torch.Tensor, map_size: torch.Tensor) -> torch.Tensor:
    bce_loss = nn.BCELoss()
    cham_loss = dist_chamfer_2D.chamfer_2DDist()
    loss = torch.zeros(()).to(cos_matrix.device)
    epsilon = 1e-8
    n = cos_matrix.shape[0]
    
    a = torch.zeros((n, n)).to(cos_matrix.device)
    for i in range(n):
        sum_a_i = torch.zeros(()).to(cos_matrix.device)
        for j in range(n):
            sum_a_i += torch.exp(cos_matrix[j][i])
        for j in range(n):
            a[i][j] = (torch.exp(cos_matrix[j][i])/sum_a_i)
            
    chamfer_idx_list = [None] * (n*n)
    
    for i in range(n):
        nodes_1 = node_map[i:i+1][:,:map_size[i],:]
        for j in range(i+1, n):
            nodes_2 = node_map[j:j+1][:,:map_size[j],:]
            dist1, dist2, idx1, idx2 = cham_loss(nodes_1, nodes_2)
            chamfer_idx_list[i*n+j] = idx1
            chamfer_idx_list[j*n+i] = idx2
    
    start = time.time()        
    for i in range(n):
        cnt = 0
        
        inner_loss = torch.zeros(()).to(cos_matrix.device)
        for v in range(int(map_size[i])):
            ws = torch.where(adj_matrixes[i][v] == 0)[0].tolist()
            for w in ws:
                if v == w:
                    continue
                cnt += 1
                rnd = torch.ones(()).to(cos_matrix.device)
                lnd = torch.zeros(()).to(cos_matrix.device)
                
                for j in range(n):
                    if j == i:
                        continue
                    idx = chamfer_idx_list[i*n+j]
                    v_index = idx[0][v]
                    w_index = idx[0][w]
                    if adj_matrixes[j][v_index][w_index] == 0:
                        lnd += a[i][j]
                    
                lnd += epsilon
                inner_loss += bce_loss(lnd, rnd)
            
        inner_loss = inner_loss / cnt if cnt != 0 else torch.zeros(())
        loss += inner_loss
    
    end = time.time()
    print("for loop time: {}".format(end - start))
    loss = loss/n
    return loss