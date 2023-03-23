from models.resnet import ModifiedResNet
import torch
from torchvision import transforms as T
import numpy as np
import PIL
import os
os.chdir("/home/zdz")
from argoverse.map_representation.map_api import ArgoverseMap
from argoverse.data_loading.argoverse_tracking_loader import ArgoverseTrackingLoader
from typing import List

camera_list = ['ring_front_center', 'ring_front_left', 'ring_front_right', 'ring_rear_left', 'ring_rear_right', 'ring_side_left', 'ring_side_right']

def fix_img(img: PIL.Image.Image) -> PIL.Image.Image:
    return img.convert('RGB') if img.mode != 'RGB' else img

if __name__ == '__main__':
    
    imgs = []
    idx = 50
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    tracking_dataset_dir = 'argoverse-api/argoverse-tracking/sample/'
    argoverse_loader = ArgoverseTrackingLoader(tracking_dataset_dir)
    argoverse_data = argoverse_loader[0]
    
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
    
    layers = [2,2,2,2]
    resnet = ModifiedResNet(layers, 512, 8, 7).to(device)
    res = resnet(img_tensors)