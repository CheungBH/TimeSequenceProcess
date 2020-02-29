import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from .src.utils import TCNData, train_val_split, get_data
from .src.model import TCN
import numpy as np

n_classes = 2
input_channels = 30
seq_length = 34
kernel_size = 5
batch_size = 8

device = "cuda:0"
steps = 0

permute = torch.Tensor(np.random.permutation(360).astype(np.float64)).long()
permute.cuda()
channel_sizes = [6, 6, 6, 6]
torch.manual_seed(1111)


class TCNTrainer:
    def __init__(self, data_path, epoch, dropout, lr, model_name, log_name):
        self.epoch = epoch
        self.model = TCN(input_channels, n_classes, channel_sizes, kernel_size=kernel_size, dropout=dropout)
        self.model.cuda()
        self.name = model_name
        self.log = open(log_name, "w")
        self.criterion = nn.CrossEntropyLoss().to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8)

        data = get_data(data_path)
        train_data, test_data = train_val_split(data)
        train_set = TCNData(train_data)
        vaild_set = TCNData(test_data)
        self.train_loader = DataLoader(train_set, batch_size=8, shuffle=True)
        self.valid_loader = DataLoader(vaild_set, batch_size=8, shuffle=True)

    def __train(self, ep):
        global steps
        train_loss = 0
        self.model.train()
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data = data.view(-1, input_channels, seq_length)
            data = data.to(device=device)
            target = target.to(device=device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, torch.max(target, 1)[1])
            loss.backward()
            self.optimizer.step()
            train_loss += loss
            steps += seq_length
            if batch_idx > 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tSteps: {}'.format(
                    ep, batch_idx * batch_size, len(self.train_loader.dataset),
                        100. * batch_idx / len(self.train_loader), train_loss.item() / 10, steps), file=self.log)

        train_loss /= len(self.train_loader.dataset)
        return train_loss

    def __test(self):
        self.model.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in self.valid_loader:
                data = data.view(-1, input_channels, seq_length)
                data, target = data.to(device=device), target.to(device=device)
                total += target.size(0)
                output = self.model(data)
                test_loss += self.criterion(output, torch.max(target, 1)[1]).item()
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(torch.max(target, 1, keepdim=True)[1]).sum().item()
            test_loss /= len(self.valid_loader.dataset)
            print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
                test_loss, correct, total,
                100. * correct / total), file=self.log)
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
    TCNTrainer("data", 100, 0.05, 1e-4, "wtf.pth", "wtf.txt").train_tcn()
