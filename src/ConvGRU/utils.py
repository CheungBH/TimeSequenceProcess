import os
import torch
import numpy as np
from torch import nn
import torch.utils.data as data
from torch.utils.data import DataLoader
from config import config

kps = config.kps_num
frm = config.training_frame


class ConvGRULoader(data.Dataset):
    def __init__(self, datas, labels, maxlabel):
        self.datas = datas
        self.labels = labels
        self.max = maxlabel

    def __len__(self):
        return len(self.datas)
    
    def __getitem__(self, idx):
        keypoint = np.asarray(self.datas[idx]).astype(np.float32).reshape(frm,int(kps/2), 2)
        keypoint = torch.from_numpy(keypoint).unsqueeze(1)#(30,1,17,2)
        label = self.labels[idx]
        label = torch.tensor(label)
        
        return keypoint, label


if __name__ == '__main__':
    datas = [[0]*30*34 for _ in range(5)]
    labels = [0,0,1,1,0]
    input_channels =(1,17,2)
    dataset = ConvGRUData(datas,labels, 2)
    dataloader = DataLoader(dataset)
    dataiter = iter(dataloader)
    data, label = dataiter.next()
    print(data.shape)
    
