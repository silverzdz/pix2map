from models.transformer import Transformer
import torch
import numpy as np
import os
os.chdir("/home/zdz")
from argoverse.map_representation.map_api import ArgoverseMap
from argoverse.data_loading.argoverse_tracking_loader import ArgoverseTrackingLoader
from typing import List

def get_vector(centerlines, x, y):
    nodes = []
    lines = []
    for centerline in centerlines:
        line = []
        for point in centerline:
            if (point[0]-x <= -50) or (point[0]-x >= 50):
                continue
            if (point[1]-y <= -50) or (point[1]-y >= 50):
                continue
            node = ("{:.0f}".format(point[0]-x), "{:.0f}".format(point[1]-y))
            nodes.append(node)
            line.append(node)
        lines.append(line)
    return nodes, lines

def cal_adjacent_matrix(centerlines, x, y):
    # centerlines shape: [lines_num, 10, 3]
    nodes, lines = get_vector(centerlines, x, y)
    node_map = {}
    node_id = 0
    for node in nodes:
        if node in node_map.keys():
            continue
        node_map[node] = node_id
        node_id += 1
    node_num = node_id
    
    # adj_matrix = np.full((512, 512), -np.inf)
    adj_matrix = np.full((node_num, node_num), -np.inf)
    for line in lines:
        line_length = len(line)
        
        first_node = line[0]
        first_id = node_map[first_node]
        adj_matrix[first_id][first_id] = 0
        
        last_node = line[-1]
        last_id = node_map[last_node]
        adj_matrix[last_id][last_id] = 0
        
        for i in range(1, line_length-1):
            node = line[i]
            node_id = node_map[node]
            if i == 1:
                adj_matrix[first_id][node_id] = 0
            if i == line_length - 2:
                adj_matrix[last_id][node_id] = 0
            pre_id = node_map[line[i-1]]
            next_id = node_map[line[i+1]]
            adj_matrix[node_id][node_id] = 0 # self
            adj_matrix[node_id][pre_id] = 0 # pre
            adj_matrix[node_id][next_id] = 0 # next
    
    return node_map, adj_matrix  

def position_encoding(node_map):
    new_map = {}
    for i in node_map.keys():
        x = int(i[0]) + 50
        y = int(i[1]) + 50
        id = node_map[i]
        encoding = y * 100 + x
        new_map[encoding] = id
    return new_map

def vectorize(encoding_map, device):
    vector_list = []
    for k in encoding_map.keys():
        v = encoding_map[k]
        tmp_vec = torch.zeros(size = (1, 512)).to(device)
        tmp_vec[:,v] = k
        vector_list.append(tmp_vec)
    return vector_list

if __name__ == '__main__':
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    am = ArgoverseMap()
    
    tracking_dataset_dir = 'argoverse-api/argoverse-tracking/sample/'
    argoverse_loader = ArgoverseTrackingLoader(tracking_dataset_dir)
    argoverse_data = argoverse_loader[0]
    
    x,y,_ = argoverse_data.get_pose(100).translation
    local_centerlines = am.find_local_lane_centerlines(x,y, "PIT", query_search_range_manhattan=40)
    
    node_map, adj_matrix = cal_adjacent_matrix(local_centerlines, x, y)
    
    encoding_map = position_encoding(node_map)
    n = len(encoding_map)
    
    
    ### using nn.Embedding to encode the original node position with padding
    encoding_tensor = torch.zeros(size = (1, 512))
    for i in range(n):
        encoding_tensor[:,i] = list(encoding_map)[i]
    encoding_tensor = encoding_tensor.int()
    
    embedding = torch.nn.Embedding(10000,512,padding_idx=0)
    input_embedding = embedding(encoding_tensor).to(device)
    
    #print(encoding_map)
    
    test_vectors = torch.zeros(size = (1, n, 512))
    for i in range(n):
        test_vectors[:,i,i] = torch.Tensor([list(encoding_map)[i]])
    
    ### TODO: how to pad vectors to 512-dimension?

    
    test_vectors = test_vectors.to(device)
    
    ### TODO: finished. attn_mask is adj_matrix
    ### how to send attention_mask ???
    ### key_padding_mask or attn_mask ???
    
    # adj_matrix[-1,:] = -np.inf
    
    num_heads = 512 // 64
    batch_size = 1
    attn_mask = np.zeros(shape = (num_heads*batch_size, adj_matrix.shape[0], adj_matrix.shape[1]))
    for i in range(num_heads*batch_size):
        attn_mask[i] = adj_matrix
    
    attn_mask = torch.Tensor(attn_mask)
    attn_mask.to(device)
    
    '''
    key_padding_mask = np.zeros(shape = (1,245))
    key_padding_mask[0][-1] = -np.inf
    key_padding_mask = torch.Tensor(key_padding_mask).bool().to(device)
    '''
    
    transformer = Transformer(512, 7, 512//64, attn_mask)
    transformer = transformer.to(device)
    
    #vec_list = vectorize(encoding_map, device)
    
    '''
    adj_list: List[List[torch.Tensor]] = []
    for i in range(n):
        v_list: List[torch.Tensor] = []
        for j in range(n):
            if adj_matrix[i][j] == 0:
                #v_list.append(test_vectors[0][j:j+1])
                v_list.append(test_vectors[0][j])
        adj_list.append(v_list)
    '''
    
    ### TODO: why zero-tensor send to transformer would get NaN?
    
    # batch size can only be 1
    # res = transformer(input_embedding)
    res = transformer(test_vectors)