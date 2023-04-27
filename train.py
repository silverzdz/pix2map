import torch
from torchvision import transforms as T
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim import Adam
from torchvision.models import ResNet18_Weights
import numpy as np
import PIL
import os
import sys
import time
from typing import List, Tuple, Dict
sys.path.append('./..')
from map.utils import get_vector, cal_adjacent_matrix, position_encoding
from models.IG_clip import ImageGraphClip
from argoverse.map_representation.map_api import ArgoverseMap
from argoverse.data_loading.argoverse_tracking_loader import ArgoverseTrackingLoader

from data.dataset import MyDataset
from models.loss import contrastive_loss, chamfer_loss_simple, edge_loss_simple


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    argoverse_dataset = MyDataset("train")
    test_dataset = MyDataset("test")
    
    n_samples = int(0.2*len(argoverse_dataset))
    indices = torch.randperm(len(argoverse_dataset))[:n_samples]
    sampler = SubsetRandomSampler(indices)
    dataloader = DataLoader(dataset=argoverse_dataset, batch_size=16, sampler=sampler, num_workers=0)
    
    test_n_samples = int(0.1*len(test_dataset))
    test_indices = torch.randperm(len(test_dataset))[:test_n_samples]
    test_sampler = SubsetRandomSampler(test_indices)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=16, sampler=test_sampler, num_workers=0)
    
    clip = ImageGraphClip(512, 224, [2,2,2,2], 7, 512, 8, 7).to(device)
    model_dict = clip.state_dict()
    
    pretrained_dict = torch.load("pretrain/resnet18-f37072fd.pth")
    for name, param in pretrained_dict.items():
        if name == "fc.weight" or "fc.bias":
            continue
        model_dict["resnet."+name].copy_(param)
    clip.load_state_dict(model_dict)
    
    optimizer = Adam(clip.parameters(), lr = 2e-4)
    
    start_epoch = 0
    num_epochs = 40
    best_loss = float("inf")
    
    # Load checkpoint if available
    checkpoint_path = "checkpoint.pth"
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        start_epoch = checkpoint["epoch"]
        clip.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        best_loss = checkpoint["best_loss"]
        print("Checkpoint loaded from epoch {} with best loss {}".format(start_epoch, best_loss))
        f = open("log.txt", "a")
        f.write("Checkpoint loaded from epoch {} with best loss {}\n".format(start_epoch, best_loss))
        f.close()
    
    for epoch in range(start_epoch, num_epochs):
        clip.train()
        
        start = time.time()
        train_loss = 0
        for idx, data in enumerate(dataloader, 0):
            if idx == len(dataloader) - 1:
                continue
            
            print("Batch {} data loaded. Start training.".format(idx))
            f = open("log.txt", "a")
            f.write("Batch {} data loaded. Start training.\n".format(idx))
            f.close()
            
            for key in data:
                data[key] = data[key].to(device)
            
            res = clip(data['img'], data['graph'], data['adj_matrix'])
            
            optimizer.zero_grad()
            
            loss1 = contrastive_loss(res)
            print("Loss1: {}".format(loss1))
            loss2 = chamfer_loss_simple(res, data['node_map'], data['map_size'])
            print("Loss2: {}".format(loss2))
            loss3 = edge_loss_simple(res, data['node_map'], data['adj_matrix'], data['map_size'])
            print("Loss3: {}".format(loss3))
            f = open("log.txt", "a")
            f.write("Loss1: {}\n Loss2: {}\n Loss3: {}\n".format(loss1, loss2, loss3))
            f.close()
            
            loss = loss1 + loss2 + 0.1*loss3
            #print("Final loss: {}".format(loss))
            train_loss += loss.item()
            
            loss.backward()
            optimizer.step()
            
            print("Epoch={}/{}, {}/{} of train, final loss={}".format(
                epoch, 40, idx, len(dataloader), loss.item()))
            
            f = open("log.txt", "a")
            f.write("Epoch={}/{}, {}/{} of train, final loss={}\n".format(
                epoch, 40, idx, len(dataloader), loss.item()))
            
            end = time.time()
            print("Batch time: {}".format(end - start))
            f.write("Batch time: {}\n".format(end - start))
            
            print("============================================\n")
            f.write("============================================\n\n")
            f.close()
            start = time.time()
        
        if len(dataloader) > 1:
            train_loss = train_loss/(len(dataloader) - 1)
        
        clip.eval()
        total_loss = 0.0
        start = time.time()
        with torch.no_grad():
            for idx, data in enumerate(test_dataloader):
                if idx == len(test_dataloader)-1:
                    continue
                for key in data:
                    data[key] = data[key].to(device)
                
                res = clip(data['img'], data['graph'], data['adj_matrix'])
                loss1 = contrastive_loss(res)
                loss2 = chamfer_loss_simple(res, data['node_map'], data['map_size'])
                loss3 = edge_loss_simple(res, data['node_map'], data['adj_matrix'], data['map_size'])
                loss = loss1 + loss2 + 0.1*loss3
                total_loss += loss.item()
                print("Epoch={}/{}, {}/{} of train, test loss={}".format(
                epoch, 40, idx, len(test_dataloader), loss.item()))
                # print("============================================\n")
                f = open("log.txt", "a")
                f.write("Epoch={}/{}, {}/{} of train, test loss={}\n".format(
                epoch, 40, idx, len(test_dataloader), loss.item()))
                # f.write("============================================\n\n")
                f.close()
        
        avg_loss = 0
        if len(test_dataloader) > 1:
            avg_loss = total_loss / (len(test_dataloader)-1)

        print("Epoch {}/{} evaluation loss: {:.4f}".format(epoch, num_epochs, avg_loss))
        end = time.time()
        print("Eval time: {}".format(end - start))
        f = open("log.txt", "a")
        f.write("Epoch {}/{} evaluation loss: {:.4f}\n".format(epoch, num_epochs, avg_loss))
        f.write("Eval time: {}\n".format(end - start))
        f.close()
        
        # Save best model based on validation loss
        if train_loss < best_loss:
            best_loss = train_loss
            torch.save(clip.state_dict(), "best_model.pth")
        
        # Save checkpoint at the end of each epoch
        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": clip.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_loss": best_loss
        }
        torch.save(checkpoint, checkpoint_path)
        print("Checkpoint saved at epoch {}\n".format(epoch+1))
        f = open("log.txt", "a")
        f.write("Checkpoint saved at epoch {}\n\n".format(epoch+1))
        f.close()