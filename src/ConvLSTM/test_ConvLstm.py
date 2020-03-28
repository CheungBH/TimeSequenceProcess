import torch
import sys
sys.path.append("../../")
from src.ConvLSTM.model import ConvLSTM
import numpy as np
import os
from config import config

device = config.device
ConvLSTM_params = config.ConvLSTM_structure
kps_num = config.test_kps_num


class ConvLSTMPredictor(object):
    def __init__(self, model_name, n_classes):
        # self.input_data = np.loadtxt(data_pth).astype(np.float32).reshape(-1,1,17,2)#(seq_l, 1, 17, 2)
        structure_num = int((model_name.split("/")[-1]).split('_')[1][6:])
        [hidden_channel, kernel_size, attention] = ConvLSTM_params[structure_num]
        self.model = ConvLSTM(input_size=(int(kps_num/2), 2),
                             input_dim=1,
                             hidden_dim=hidden_channel,
                             kernel_size=kernel_size,
                             num_layers=len(hidden_channel),
                             num_classes=n_classes,
                             batch_size=2,
                             batch_first=True,
                             bias=True,
                             return_all_layers=False,
                             attention=attention)
        self.model.load_state_dict(torch.load(model_name))
        if device != "cpu":
            self.model.cuda()
        self.model.eval()

    def get_input_data(self, input_data):
        data = torch.from_numpy(input_data)
        data = data.unsqueeze(0).to(device=device)
        return data#(1, 30, 1, 17, 2)

    def predict(self, data):
        input = self.get_input_data(data.reshape(-1,1,int(kps_num/2),2))
        output = self.model(input)
        #print('output:',output)
        pred = output.data.max(1, keepdim=True)[1]
        #print('pred:',pred)
        return pred


if __name__ == '__main__':
    model = 'ConvLSTM_struct2_10-10-10-10-10-99.pth'
    input_pth = 'data.txt'
    inp = np.loadtxt(input_pth).astype(np.float32).reshape(-1,1,17,2)
    prediction = ConvLSTMPredictor(model, 2)
    res = prediction.predict(inp)
    print(res)

