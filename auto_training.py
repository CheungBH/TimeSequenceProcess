from models.TCN.train_TCN import TCNTrainer
from config import config
import time
import os

model_dest = "tmp/test"
os.makedirs(os.path.join(model_dest, "model"), exist_ok=True)
os.makedirs(os.path.join(model_dest, "log"), exist_ok=True)
res = open(os.path.join(model_dest, "result.txt"), "w")


if __name__ == '__main__':
    cnt = 0
    for net in config.networks:
        for epoch in config.epoch_ls[net]:
            for dropout in config.dropout_ls[net]:
                for lr in config.lr_ls[net]:
                    cnt += 1
                    print("Begin training network {}".format(cnt))
                    print("Begin training {}: {} epochs, {} dropout, {} lr".format(net, epoch, dropout, lr))
                    time_str = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
                    begin_time = time.time()
                    if net == "TCN":
                        model_name = os.path.join(model_dest, "model", "TCN_" + time_str + ".pth")
                        log_name = os.path.join(model_dest, "log", "TCN_" + time_str + ".txt")
                        train_loss, val_loss = TCNTrainer(config.data_path, epoch, dropout, lr, model_name, log_name).train_tcn()
                        cost = time.time() - begin_time
                        print("Total time cost is {}s\n".format(cost))
                        res.write("{}: \n{} epochs, {} learning-rate, {} dropout\n".format("TCN_" + time_str + ".pth", epoch, lr, dropout))
                        res.write("Min train loss is {}. Min val loss is {}\n\n".format(train_loss, val_loss))
