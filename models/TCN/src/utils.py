#!/usr/bin/env python
# coding: utf-8
import os
from sklearn.model_selection import train_test_split
import torch
import numpy as np
from torch import nn
import torch.utils.data as data
from torch.utils.data import DataLoader

data_pth = './data'
label = ['drown', 'normal']
output_pth = 'data.txt'


def get_data(data_pth):
    file_list = []
    for parent, dirname, filenames in os.walk(data_pth):
        for filename in filenames:
            #print('parent:', parent)
            #print('filenames:', filenames)
            curr_file = parent.split(os.sep)[-1]
            #print(curr_file)
            if curr_file == 'drown_n':
                labels = 0
            if curr_file == 'normal':
                labels = 1
            parent_dir = os.path.join(data_pth,curr_file)
            file_list.append([os.path.join(parent_dir, filename), labels])
    return file_list


def write_txt(file_list, output_pth):
    with open(output_pth, 'a') as f:
        for line in file_list:
            str_line = ""
            for col, data in enumerate(line):
                if col != len(line)-1:
                    str_line = str_line + str(data) + " "
                else:
                    str_line = str_line + str(data) + "\n"
            f.write(str_line)


def train_val_split(datas):
    train_data, test_data = train_test_split(datas, test_size=0.2, random_state=42, shuffle=True)
    return train_data, test_data


class TCNData(data.Dataset):
    def __init__(self, datas):
        self.datas = datas
        self.kp = []
        self.labels = []
        for data in self.datas:
            kp, label = data[0], data[1]
            self.kp.append(kp)
            self.labels.append(label)
    
    def get_one_hot_num(self, label):
        if label == 0:
            return torch.tensor([1,0])
        
        if label == 1:
            return torch.tensor([0,1])

    def __len__(self):
        return len(self.datas)
    
    @property
    def get_keypoints_shape(self):
        return len(self.kp)
    @property
    def get_labels(self):
        return self.labels
    
    def __getitem__(self, idx):
        keypoint = np.loadtxt(self.kp[idx]).astype(np.float32) #reshape(34, -1).[:,:30]

        if keypoint.shape[0]<30:
            tmp = np.zeros((30-keypoint.shape[0],34)).astype(np.float32)
            keypoint = np.concatenate((keypoint,tmp),axis=0)

        seq = torch.from_numpy(keypoint)#.permute(1,0)
        label = self.labels[idx]
        label = self.get_one_hot_num(label)
        return (seq, label)

    
if __name__ == '__main__':
    data = get_data(data_pth)
    #write_txt(data, output_pth)
    dataset = TCNData(data)
    #print(dataset.get_labels)
    dataloader = DataLoader(dataset)
    dataiter = iter(dataloader)
    data, label = dataiter.next()
    print(data.shape)
    #print(label.data.max(1,keepdim=True)[1])
