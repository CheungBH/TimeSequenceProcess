import torch
import sys
sys.path.append("../../")
from src.ConvLSTM.model import ConvLSTM
import numpy as np
import os
from config import config

device = config.device
params_ls = config.ConvLSTM_structure


class ConvLSTMPredictor(object):
    def __init__(self, model_name):
        # self.input_data = np.loadtxt(data_pth).astype(np.float32).reshape(-1,1,17,2)#(seq_l, 1, 17, 2)
        structure_num = int(model_name.split('_')[1][6:])
        self.model = ConvLSTM(input_size=(17, 2),
                             input_dim=1,
                             hidden_dim=params_ls[structure_num][0],
                             kernel_size=params_ls[structure_num][1],
                             num_layers=len(params_ls[structure_num][0]),
                             num_classes=2,
                             batch_size=2,
                             batch_first=True,
                             bias=True,
                             return_all_layers=False,
                             attention=params_ls[structure_num][2])
        self.model.load_state_dict(torch.load(model_name, map_location=device))
        if device != "cpu":
            self.model.cuda()
        self.model.eval()

    def get_input_data(self, input_data):
        data = torch.from_numpy(input_data)
        data = data.unsqueeze(0).to(device=device)
        return data#(1, 30, 1, 17, 2)

    def predict(self, data):
        input = self.get_input_data(data.reshape(-1,1,17,2))
        output = self.model(input)
        #print('output:',output)
        pred = output.data.max(1, keepdim=True)[1]
        #print('pred:',pred)
        return pred


if __name__ == '__main__':
    model = 'ConvLSTM_struct2_10-10-10-10-10-99.pth'
    input_pth = 'data.txt'
    inp = np.loadtxt(input_pth).astype(np.float32).reshape(-1,1,17,2)
    prediction = ConvLSTMPredictor(model)
    res = prediction.predict(inp)
    print(res)

