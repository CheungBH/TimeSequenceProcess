import torch
import sys
sys.path.append("../../")
from models.TCN.src.model import TCN
import numpy as np
from config import config


class TCNPredictor(object):
    def __init__(self, model):
        # self.input_data = np.loadtxt(data_pth).astype(np.float32).reshape(-1,34)#(seq_l, 34)
        self.model_pth = model
        self.model = torch.load(self.model_pth)
        self.model.eval()

    def get_input_data(self, input_data):
        data = torch.from_numpy(input_data)
        data = data.unsqueeze(0).to(device=config.device)#(1, 30, 34)
        data = data.permute(0,2,1)
        return data#(1, 34, 30)

    def predict(self, data):
        input = self.get_input_data(data)
        # input = input.type(torch.cuda.FloatTensor)
        output = self.model(input)
        #print('output:',output)
        pred = output.data.max(1, keepdim=True)[1]
        #print('pred:',pred)
        return pred

    # def predict_all(self):
    #     preds = []
    #     for i in range(0, self.input_data.shape[0], 30):
    #         data = self.get_input_data(self.input_data[i:i+30,:])
    #         if data.size(1)<30:
    #             break
    #         pred = self.predict_pre_second(data)
    #         print('pred:',pred)
    #         preds.append(pred)
    #     return preds


if __name__ == '__main__':
    model_pth = 'TCN.pth'
    input_pth = 'data.txt'
    inp = np.loadtxt(input_pth).astype(np.float32).reshape(-1,34)
    prediction = TCNPredictor(model_pth)
    prediction.predict(inp)