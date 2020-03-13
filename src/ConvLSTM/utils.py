import os
import torch
import numpy as np
from torch import nn
import torch.utils.data as data
from torch.utils.data import DataLoader
from config import config

kps = config.kps_num
frm = config.training_frame


class ConvLstmLoader(data.Dataset):
    def __init__(self, datas, labels, maxlabel):
        self.datas = datas#[N, 30*34]
        self.labels = labels#[N, 1]
        self.max = maxlabel
        self.vector = [0]*self.max
    
    def get_one_hot_num(self, label):
        # self.vector[label] = 1
        # one_hot_vextor = self.vector
        # self.vector = [0]*self.max
        # return torch.LongTensor(one_hot_vextor)
        return torch.tensor(label)
    
    def __len__(self):
        return len(self.datas)
    
    def __getitem__(self, idx):
        keypoint = np.asarray(self.datas[idx]).astype(np.float32).reshape(frm, int(kps/2), 2)#[30*34]#.reshape(-1,1,17,2)
        # keypoint = keypoint[:,np.newaxis]#[30*34,1]
        keypoint = torch.from_numpy(keypoint).unsqueeze(1)
        one_hot_label = self.get_one_hot_num(self.labels[idx])

        return (keypoint, one_hot_label)


if __name__ == '__main__':
    datas = [[0]*30*34 for _ in range(6)]
    #print(len(datas[0]))
    labels = [0,0,1,1,0,1]
    input_channels = (1,17,2)
    dataset = ConvLstmLoader(datas, labels, 2)
    dataloader = DataLoader(dataset)
    dataiter = iter(dataloader)
    data, label = dataiter.next()
    data = data.view(-1, 30, input_channels[0], input_channels[1], input_channels[2])
    print(data.shape)
