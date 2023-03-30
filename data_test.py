from models.IG_clip import ImageGraphClip
import torch
from torchvision import transforms as T
import numpy as np
import os
import time
import PIL
os.chdir("/home/zdz")
from argoverse.map_representation.map_api import ArgoverseMap
from argoverse.data_loading.argoverse_tracking_loader import ArgoverseTrackingLoader
from typing import List
from map.utils import get_vector, cal_adjacent_matrix, position_encoding
from models.loss import contrastive_loss, chamfer_loss, edge_loss

camera_list = ['ring_front_center', 'ring_front_left', 'ring_front_right', 'ring_rear_left', 'ring_rear_right', 'ring_side_left', 'ring_side_right']

def fix_img(img: PIL.Image.Image) -> PIL.Image.Image:
    return img.convert('RGB') if img.mode != 'RGB' else img

if __name__ == '__main__':
    
    batch_size = 16
    num_heads = 512 // 64
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    idx = 0
    
    image_transform = T.Compose([
        T.Lambda(fix_img),
        T.RandomResizedCrop(224,
                            scale = (0.75 , 1.),
                            ratio=(1., 1.)),
        
        T.ToTensor(),
        T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ])
    
    am = ArgoverseMap()
    
    tracking_dataset_dir = '/mnt/d/dataset/argoverse/argoverse-tracking/train1/'
    argoverse_loader = ArgoverseTrackingLoader(tracking_dataset_dir)
    
    
    img_input = torch.zeros((1, 512))
    graph_input = torch.zeros((1, 512))
    attn_mask = np.zeros(shape = (num_heads*batch_size, 512, 512))
    
    node_maps = []
    adj_matrixes = []
    
    for scene in range(batch_size):
        idx = 0
        argoverse_data = argoverse_loader[scene]
        
        x,y,_ = argoverse_data.get_pose(idx).translation
        local_centerlines = am.find_local_lane_centerlines(x,y, argoverse_data.city_name, query_search_range_manhattan=40)
        
        # print("x:{}, y:{}".format(x, y))
    
        ### images
        imgs = []
        for camera in camera_list:
            img = argoverse_data.get_image_sync(idx, camera = camera)
            imgs.append(img)
            
        imgs = [PIL.Image.fromarray(img) for img in imgs]
        
        img_tensors = [image_transform(img).unsqueeze(0).to(device) for img in imgs]
        img_tensor = torch.stack(img_tensors, dim=1)
        if scene == 0:
            img_input = img_tensor
        else:
            img_input = torch.cat([img_input, img_tensor], dim=0)
    
        ### graphs
        node_map, adj_matrix = cal_adjacent_matrix(local_centerlines, x, y)
        node_maps.append(node_map)
        adj_matrixes.append(adj_matrix)
        encoding_map = position_encoding(node_map)
        n = len(encoding_map)
        
        encoding_tensor = torch.zeros(size = (1, 512))
        for i in range(n):
            encoding_tensor[:,i] = list(encoding_map)[i]
        encoding_tensor = encoding_tensor.int().to(device)
        if scene == 0:
            graph_input = encoding_tensor
        else:
            graph_input = torch.cat([graph_input, encoding_tensor], dim=0)
    
        ### attn_mask
        for i in range(8):
            attn_mask[scene*8+i] = adj_matrix
        
    attn_mask = torch.Tensor(attn_mask).to(device)
    #attn_mask.to(device)
    
    ### clip
    clip = ImageGraphClip(512, 224, [2,2,2,2], 7, 512, 8, 7, attn_mask).to(device)
    
    res = clip(img_input, graph_input)
    
    loss1 = contrastive_loss(res)
    loss2 = chamfer_loss(res, node_maps)
    
    ### whether loss3 times 0.1?
    loss3 = edge_loss(res, node_maps, adj_matrixes)