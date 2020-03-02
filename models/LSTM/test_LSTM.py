import numpy as np
from keras.models import load_model

# input_pth = 'data/AlphaPose_drown_34.txt'
# input_data = np.loadtxt(input_pth)
# input_data = input_data[np.newaxis,]
#
#
# model = load_model("model.h5")
# output = model.predict(input_data)
# pred = np.argmax(output)
# print(pred)


def test_LSTM(input_data, model_path):
    model = load_model(model_path)
    output = model.predict(input_data)
    pred = np.argmax(output)
    return pred

