import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from models.TCN.src.utils import TCNLoader
from models.TCN.src.model import TCN
import numpy as np
import os
from sklearn.model_selection import train_test_split
from config import config


input_channels = config.kps_num
log_interval = config.log_interval
seq_length = config.training_frame

device = "cuda:0"
steps = 0

permute = torch.Tensor(np.random.permutation(360).astype(np.float64)).long()
permute.cuda()
channel_sizes = [6, 6, 6, 6]
kernel_size = 5
torch.manual_seed(1111)


class TCNTrainer:
    def __init__(self, data_path, epoch, dropout, lr, model_name, log_name, batch_size, n_classes):
        self.epoch = epoch
        self.batch_size = batch_size
        self.model = TCN(input_channels, n_classes, channel_sizes, kernel_size=kernel_size, dropout=dropout)
        self.model.cuda()
        self.name = model_name
        self.log = open(log_name, "w")
        self.criterion = nn.NLLLoss().to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8)

        self.data, self.label = self.__load_data(data_path)
        sample = [(d, l) for d, l in zip(self.data, self.label)]
        train_sample, test_sample = train_test_split(sample, test_size=0.2, random_state=42, shuffle=True)
        train_data, train_labels = self.__separate_sample(train_sample)
        test_data, test_labels = self.__separate_sample(test_sample)
        train_set = TCNLoader(train_data, train_labels, n_classes)
        vaild_set = TCNLoader(test_data, test_labels, n_classes)
        self.train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True)
        self.valid_loader = DataLoader(vaild_set, batch_size=self.batch_size, shuffle=True)

    def __load_data(self, root):
        with open(os.path.join(root, "data.txt"), "r") as data_f:
            data = []
            for line in data_f.readlines():
                origin_ls = self.__ls_preprocess(line.split("\t"))
                ls = [float(item) for item in origin_ls]
                data.append(np.array(ls).reshape((seq_length, input_channels)))
            data_f.close()

        with open(os.path.join(root, "label.txt"), "r") as label_f:
            label = [int(line[:-1]) for line in label_f.readlines()]
            label_f.close()
        return data, label

    @staticmethod
    def __separate_sample(sample):
        data, label = [], []
        for item in sample:
            data.append(item[0])
            label.append(item[1])
        return data, label

    @staticmethod
    def __ls_preprocess(ls):
        try:ls.remove("\n")
        except: pass
        while True:
            try: ls.remove("")
            except ValueError: break
        return ls

    def __train(self, ep):
        global steps
        train_loss = 0
        self.model.train()
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data = data.permute(0, 2, 1)
            data = data.to(device=device)
            target = target.to(device=device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            train_loss += loss
            steps += seq_length
            if batch_idx > 0 and batch_idx % log_interval == 0:
                out_log = 'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tSteps: {}'.format(
                    ep, batch_idx * self.batch_size, len(self.train_loader.dataset),
                        100. * batch_idx / len(self.train_loader), train_loss.item() / log_interval, steps)
                print(out_log)
                self.log.write(out_log + "\n")

        train_loss /= len(self.train_loader.dataset)
        return train_loss

    def __test(self):
        self.model.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in self.valid_loader:
                data = data.permute(0, 2, 1)
                data, target = data.to(device=device), target.to(device=device)
                total += target.size(0)
                output = self.model(data)
                test_loss += self.criterion(output, target).item()
                pred = output.data.max(1)[1]
                correct += pred.eq(target).sum().item()
            test_loss /= len(self.valid_loader.dataset)
            out_log = 'Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                test_loss, correct, total,
                100. * correct / total)
            print(out_log)
            self.log.write(out_log + "\n")
        test_loss /= len(self.valid_loader.dataset)
        return test_loss

    def train_tcn(self):
        min_val_loss, min_train_loss = float("inf"), float("inf")
        for epoch in range(1, self.epoch + 1):
            train_loss = self.__train(epoch)
            min_train_loss = train_loss if train_loss < min_train_loss else min_train_loss
            val_loss = self.__test()
            if val_loss < min_val_loss:
                min_val_loss = val_loss
                torch.save(self.model, self.name)
        return min_train_loss, min_val_loss


if __name__ == '__main__':
    TCNTrainer("../../tmp/input1", 1000, 0.05, 1e-4, "wtf.pth", "wtf.txt", 32).train_tcn()
