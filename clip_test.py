from models.IG_clip import ImageGraphClip
import torch
from torchvision import transforms as T
import numpy as np
import os
import PIL
os.chdir("/home/zdz")
from argoverse.map_representation.map_api import ArgoverseMap
from argoverse.data_loading.argoverse_tracking_loader import ArgoverseTrackingLoader
from typing import List
from map.utils import get_vector, cal_adjacent_matrix, position_encoding

camera_list = ['ring_front_center', 'ring_front_left', 'ring_front_right', 'ring_rear_left', 'ring_rear_right', 'ring_side_left', 'ring_side_right']

def fix_img(img: PIL.Image.Image) -> PIL.Image.Image:
    return img.convert('RGB') if img.mode != 'RGB' else img

if __name__ == '__main__':
    
    batch_size = 16
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    imgs = []
    idx = 50
    
    am = ArgoverseMap()
    
    tracking_dataset_dir = 'argoverse-api/argoverse-tracking/sample/'
    argoverse_loader = ArgoverseTrackingLoader(tracking_dataset_dir)
    argoverse_data = argoverse_loader[0]
    
    x,y,_ = argoverse_data.get_pose(idx).translation
    local_centerlines = am.find_local_lane_centerlines(x,y, "PIT", query_search_range_manhattan=40)
    
    ### images
    for camera in camera_list:
        img = argoverse_data.get_image_sync(idx, camera = camera)
        imgs.append(img)
        
    imgs = [PIL.Image.fromarray(img) for img in imgs]
    
    image_transform = T.Compose([
        T.Lambda(fix_img),
        T.RandomResizedCrop(224,
                            scale = (0.75 , 1.),
                            ratio=(1., 1.)),
        
        T.ToTensor(),
        T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ])
    
    img_tensors = [image_transform(img).unsqueeze(0).to(device) for img in imgs]
    img_tensor = torch.stack(img_tensors, dim=1)
    
    img_tensor = torch.cat([img_tensor]*batch_size, dim=0)
    
    ### graphs
    node_map, adj_matrix = cal_adjacent_matrix(local_centerlines, x, y)
    encoding_map = position_encoding(node_map)
    n = len(encoding_map)
    
    encoding_tensor = torch.zeros(size = (1, 512))
    for i in range(n):
        encoding_tensor[:,i] = list(encoding_map)[i]
    encoding_tensor = encoding_tensor.int()
    graph_tensor = torch.cat([encoding_tensor]*batch_size, 0).to(device)
    
    ### attn_mask
    num_heads = 512 // 64
    attn_mask = np.zeros(shape = (num_heads*batch_size, adj_matrix.shape[0], adj_matrix.shape[1]))
    for i in range(num_heads*batch_size):
        attn_mask[i] = adj_matrix
    
    attn_mask = torch.Tensor(attn_mask)
    attn_mask.to(device)
    
    ### clip
    clip = ImageGraphClip(512, 224, [2,2,2,2], 7, 512, 8, 7, attn_mask).to(device)
    
    res = clip(img_tensor, graph_tensor)