import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
import numpy as np
import PIL
import sys
import time
from typing import List, Tuple, Dict
sys.path.append('./..')
from map.utils import get_vector, cal_adjacent_matrix, position_encoding
from models.IG_clip import ImageGraphClip
from argoverse.map_representation.map_api import ArgoverseMap
from argoverse.data_loading.argoverse_tracking_loader import ArgoverseTrackingLoader
from models.loss import contrastive_loss, chamfer_loss, edge_loss, chamfer_loss_simple, edge_loss_simple

camera_list = ['ring_front_center', 'ring_front_left', 'ring_front_right', 'ring_rear_left', 'ring_rear_right', 'ring_side_left', 'ring_side_right']

root_dir = '/home/zdz/dataset/argoverse/argoverse-tracking/'
dir_name = {'train': ['train1', 'train2', 'train3', 'train4'], 'test': ['test']}

def fix_img(img: PIL.Image.Image) -> PIL.Image.Image:
    return img.convert('RGB') if img.mode != 'RGB' else img

image_transform = T.Compose([
        T.Lambda(fix_img),
        T.RandomResizedCrop(224,
                            scale = (0.75 , 1.),
                            ratio=(1., 1.)),
        
        T.ToTensor(),
        T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ])

### TODO: How to pass attn_mask? Might be initialized in forward().
class MyDataset(Dataset):
    def __init__(self, flag: str, data_root: str = root_dir, device: torch.device = torch.device("cpu")) -> None:
        super().__init__()
        # 0: img
        # 1: graph
        # 2: adj_matrix
        # 3: meta_info
        assert flag in ['train', 'test']
        self.flag = flag
        self.data: List[Dict] = []
        self.device = device
        self.index_array, self.max_num = self.get_index(flag, data_root)
        self.am = ArgoverseMap()
        # self.__load_data__(flag, data_root)
        
    def __getitem__(self, index: int) -> Dict:
        ### TODO: data load is implemented here, not below
        i, j, k = self.reverse_search(index)
        argoverse_loader = ArgoverseTrackingLoader(root_dir + dir_name[self.flag][i])
        argoverse_data = argoverse_loader[j]
        
        data = {}
               
        # img
        imgs = [argoverse_data.get_image_sync(k, camera) for camera in camera_list]
        imgs = [PIL.Image.fromarray(img) for img in imgs]
        data['img'] = torch.stack([image_transform(img).to(self.device) for img in imgs], dim=0)
        
        # graph & adj_mask
        x, y, _ = argoverse_data.get_pose(k).translation
        local_centerlines = self.am.find_local_lane_centerlines(x,y, argoverse_data.city_name, query_search_range_manhattan=40)
        node_map, adj_matrix = cal_adjacent_matrix(local_centerlines, x, y)
        encoding_map = position_encoding(node_map)
        encoding_tensor = torch.zeros(size = (512,))
        for k in range(len(encoding_map)):
            encoding_tensor[k] = list(encoding_map)[k]
        data['graph'] = encoding_tensor.int().to(self.device)
        data['adj_matrix'] = torch.Tensor(adj_matrix).to(self.device)
        
        # node_map
        nodes = list(node_map.keys())
        node_map = np.array([np.array(node) for node in nodes])
        map_size = node_map.shape[0]
        node_map = np.pad(node_map, ((0, 512-node_map.shape[0]), (0, 0)), 'constant', constant_values = ((0,0), (0,0)))
        
        data['node_map'] = torch.Tensor(node_map).to(self.device)
        data['map_size'] = map_size

        return data
    
    def __len__(self) -> int:
        return self.max_num
    
    def get_index(self, flag: str, data_root: str) -> Tuple[np.ndarray, int]:
        max_num = 0
        index_array = np.zeros((4,20), dtype=np.dtype(int))
        if flag == 'test':
            index_array = np.zeros((1,24), dtype=np.dtype(int))
            
        for sub_dir in dir_name[flag]:
            argoverse_loader = ArgoverseTrackingLoader(data_root+sub_dir)
            log_num = len(argoverse_loader.log_list)
            for i in range(log_num):
                data = argoverse_loader[i]
                frame_num = len(data.image_list_sync['ring_front_center'])
                max_num += frame_num
                index_array[dir_name[flag].index(sub_dir)][i] = frame_num
        return index_array, max_num
    
    def reverse_search(self, index: int) -> Tuple[int, int, int]:
        assert index >= 0 and index < self.max_num
        cnt = 0
        for i in range(self.index_array.shape[0]):
            for j in range(self.index_array.shape[1]):
                cnt = cnt + self.index_array[i][j]
                if cnt >= index:
                    return i, j, cnt-index
            
        
### Test:
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    argoverse_dataset = MyDataset('train')
    dataloader = DataLoader(dataset=argoverse_dataset, batch_size=16, shuffle=True, num_workers=0)
    
    print("Initialize finished.")
    start1 = time.time()
    for idx, data in enumerate(dataloader, 0):
        print("Loading batch {} finished.".format(idx))
        
        end1 = time.time()
        print("Loading time: {}".format(end1 - start1))
        
        for key in data:
            data[key] = data[key].to(device)
            
        clip = ImageGraphClip(512, 224, [2,2,2,2], 7, 512, 8, 7).to(device)
        
        res = clip(data['img'], data['graph'], data['adj_matrix'])
        
        loss1 = contrastive_loss(res)
        
        print("Loss1: {}".format(loss1))
        
        loss2 = chamfer_loss_simple(res, data['node_map'], data['map_size'])
        print("Loss2: {}".format(loss2))
        
        end2 = time.time()
        print("Forward + Loss1 + Loss2 time: {}".format(end2 - end1))
        
        # loss2 = chamfer_loss(res, data['node_map'], data['map_size'])

        ### whether loss3 times 0.1?
        loss3 = edge_loss_simple(res, data['node_map'], data['adj_matrix'], data['map_size'])
        
        print("Loss3: {}".format(loss3))
        
        end3 = time.time()
        print("Loss3 time: {}".format(end3 - end2))
        
        loss = loss1 + loss2 + 0.1*loss3
        print("Final loss: {}".format(loss))
        
        print("Batch time: {}".format(end3 - start1))
        
        pass