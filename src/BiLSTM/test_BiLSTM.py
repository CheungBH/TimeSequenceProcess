import torch
import sys
sys.path.append("../../")
from src.BiLSTM.model import BILSTM
import numpy as np
import os
from config import config
device = config.device
BiLSTM_params = config.BiLSTM_structure
kps_num = config.test_kps_num


class BiLSTMPredictor():
    def __init__(self, model_name, n_classes):
        struct_num = int(model_name.split('/')[-1].split('_')[1][6:])
        [hidden_dims, num_rnn_layers, attention] = BiLSTM_params[struct_num]
        self.model = BILSTM(num_classes=n_classes, input_dim=kps_num, hidden_dims=hidden_dims, num_rnn_layers=num_rnn_layers, attention=attention)
        self.model.load_state_dict(torch.load(model_name))
        if device != 'cpu':
            self.model.cuda()
        self.model.eval()
    
    def get_input_data(self, input_data):
        data = torch.from_numpy(input_data)
        data = data.unsqueeze(0).to(device=device)
        return data#(1, 30, 34)

    def predict(self, data):
        data = self.get_input_data(data.reshape(30, 34))
        output = self.model(data)
        pred = output.data.max(1, keepdim=True)[1]
        return pred


if __name__ == '__main__':
    model_name = 'BiLSTM_struct1_2020-03-13-17-49-18.pth'
    input_pth = './data/data.txt'
    inp = np.loadtxt(input_pth).astype(np.float32)
    prediction = BiLSTMPredictor(model_name,2)
    res = prediction.predict(inp)
    print(res)
    