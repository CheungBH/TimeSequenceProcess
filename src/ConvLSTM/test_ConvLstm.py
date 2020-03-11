import torch
import sys
sys.path.append("../../")
from models.ConvLSTM.model import ConvLSTM
import numpy as np
from config import config


class ConvLSTMPredictor:
    def __init__(self, model_pth):
        # self.input_data = np.loadtxt(data_pth).astype(np.float32).reshape(-1,1,17,2)#(seq_l, 1, 17, 2)
        self.model_pth = model_pth
        self.model = torch.load(self.model_pth)
        self.model.eval()

    def get_input_data(self, input_data):
        data = torch.from_numpy(input_data)
        data = data.unsqueeze(0).to(device=config.device)
        return data

    def predict(self, data):
        input = self.get_input_data(data.reshape(-1,1,17,2))
        output = self.model(input)
        pred = output.data.max(1, keepdim=True)[1]
        return pred

    # def predict_all(self):
    #     preds = []
    #     for i in range(0, self.input_data.shape[0], 30):
    #         data = self.get_input_data(self.input_data[i:i+30,:,:,:])
    #         if data.size(1)<30:
    #             break
    #         pred = self.predict_pre_second(data)
    #         print('pred:',pred)
    #         preds.append(pred)
    #     return pred


if __name__ == '__main__':
    model = './wtf.pth'
    input_pth = 'test100_0.txt'
    inp = np.loadtxt(input_pth).astype(np.float32).reshape(-1,1,17,2)
    prediction = ConvLSTMPredictor(model)
    prediction.predict(inp)
    
