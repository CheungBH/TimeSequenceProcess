from src.TCN.train_TCN import TCNTrainer
from src.LSTM.train_LSTM import LSTMTrainer
from src.ConvLSTM.train_ConvLSTM import ConvLSTMTrainer
from src.BiLSTM.train_BiLSTM import BiLSTMTrainer
from src.ConvGRU.train_ConvGRU import ConvGRUTrainer
from config import config
import time
import shutil
import os

res_dest = config.out_dest
src_data_path = config.train_data_path
if os.path.exists(res_dest):
    inp = input("The destination path '{}' exists! Continue anyway? (Press n to stop)".format(res_dest))
    if inp == "no":
        raise ValueError("The destination path '{}' exists!".format(res_dest))
os.makedirs(os.path.join(res_dest, "model"), exist_ok=True)
os.makedirs(os.path.join(res_dest, "log"), exist_ok=True)
# n_classes = len(config.training_labels)
with open(os.path.join(src_data_path, "cls.txt"), "r") as cls_file:
    n_classes = len(cls_file.readlines())
n_classes = config.num_classes_pose


if __name__ == '__main__':
    for n in config.networks:
        assert n in ['LSTM', "TCN", "ConvLSTM", "BiLSTM", "ConvGRU"], "Wrong model name '{}', please check".format(n)

    print("\n\n\n\nChecking information...")
    print("Your target networks -------> {}".format(config.networks))
    print("The output folder -------> '{}'".format(res_dest))
    print("The input data comes from --------> {}".format(src_data_path))
    nums = [len(config.epoch_ls[net])*len(config.dropout_ls[net])*len(config.lr_ls[net])*len(config.structure_ls[net])
            for net in config.networks]
    print("Total network number -------> {}".format(sum(nums)))
    input("Press any keys to continue")

    res = open(os.path.join(res_dest, "training_result.csv"), "w")
    shutil.copy(os.path.join(src_data_path, "cls.txt"), res_dest)
    with open(os.path.join("6_network", "description.txt"), "a+") as f:
        f.write("\n" + config.data_info)
    res.write("model_name, model_type, epochs, dropout, learning-rate, structure_num, min_train_loss, min_val_loss,"
              "max_val_acc\n")
    res.close()

    cnt = 0
    for net in config.networks:
        for epoch in config.epoch_ls[net]:
            for dropout in config.dropout_ls[net]:
                for lr in config.lr_ls[net]:
                    for num in config.structure_ls[net]:
                        cnt += 1
                        print("Begin training network [{}/{}]".format(cnt, sum(nums)))
                        print("Begin training {}: {} epochs, {} dropout, {} lr, struct {}".format
                              (net, epoch, dropout, lr, num))
                        time_str = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
                        begin_time = time.time()
                        batch_size = config.batch_size[net]
                        net_string = "{}_struct{}_".format(net, num) + time_str
                        res = open(os.path.join(res_dest, "training_result.csv"), "a+")
                        #try:
                        if net == "TCN":
                            log_name = os.path.join(res_dest, "log", net_string + ".txt")
                            model_name = os.path.join(res_dest, "model", net_string + ".pth")
                            train_loss, val_loss, val_acc = TCNTrainer(src_data_path, epoch, dropout, lr, model_name,
                                                log_name, batch_size, n_classes, num).train_tcn()

                        elif net == "LSTM":
                            log_name = os.path.join(res_dest, "log", net_string + ".csv")
                            model_name = os.path.join(res_dest, "model", net_string + ".h5")
                            os.makedirs(os.path.join(res_dest, "LSTM_graph/loss"), exist_ok=True)
                            os.makedirs(os.path.join(res_dest, "LSTM_graph/acc"), exist_ok=True)
                            train_loss, val_loss, val_acc = LSTMTrainer(src_data_path, epoch, dropout, lr, model_name,
                                                log_name, batch_size, n_classes, num).train_LSTM()

                        elif net == "ConvLSTM":
                            log_name = os.path.join(res_dest, "log", net_string + time_str + ".txt")
                            model_name = os.path.join(res_dest, "model", net_string + ".pth")
                            train_loss, val_loss, val_acc = ConvLSTMTrainer(src_data_path, epoch, dropout, lr, model_name,
                                                log_name, batch_size, n_classes, num).train_convlstm()

                        elif net == "BiLSTM":
                            log_name = os.path.join(res_dest, "log", net_string + time_str + ".txt")
                            model_name = os.path.join(res_dest, "model", net_string + ".pth")
                            train_loss, val_loss, val_acc = BiLSTMTrainer(src_data_path, epoch, dropout, lr, model_name,
                                                log_name, batch_size, n_classes, num).train_bilstm()

                        elif net == "ConvGRU":
                            log_name = os.path.join(res_dest, "log", net_string + time_str + ".txt")
                            model_name = os.path.join(res_dest, "model", net_string + ".pth")
                            train_loss, val_loss, val_acc = ConvGRUTrainer(src_data_path, epoch, dropout, lr, model_name,
                                                log_name, batch_size, n_classes, num).train_convgru()

                        else:
                            raise ValueError("Wrong model type")

                        cost = time.time() - begin_time
                        with open(log_name, "a+") as log:
                            print("Total time cost is {}s\n".format(cost))
                            log.write("\nTotal time cost is {}s\n".format(cost))
                        res.write("{},{},{},{},{},{},{},{},{}\n".format(
                            net_string + ".pth", net, epoch, dropout, lr, num, train_loss, val_loss, val_acc))
                        # except:
                        #     res.write("{},Error occurs when training!\n".format(net_string))
                        res.close()
