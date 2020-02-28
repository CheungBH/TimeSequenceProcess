from models.TCN.train_TCN import TCNTrainer
from config import config
import time
import os

model_dest = "tmp/test"
os.makedirs(os.path.join(model_dest, "model"), exist_ok=True)
os.makedirs(os.path.join(model_dest, "log"), exist_ok=True)
res = open(os.path.join(model_dest, "result.txt"), "w")


if __name__ == '__main__':
    for net in config.networks:
        for epoch in config.epoch_ls[net]:
            for dropout in config.dropout_ls[net]:
                for lr in config.lr_ls[net]:
                    if net == "TCN":
                        time_str = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
                        model_name = os.path.join(model_dest, "model", "TCN_" + time_str + ".pth")
                        log_name = os.path.join(model_dest, "log", "TCN_" + time_str + ".txt")
                        TCNTrainer(config.data_path, epoch, dropout, lr, model_name, log_name).train_tcn()
                        res.write("{}\n: {} epochs, {} learning-rate, {} dropout\n\n".format(model_name, epoch, lr, dropout))
