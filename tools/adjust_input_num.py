import os
from collections import Counter
import random

err = 0.05
new_folder = "equal"


class AdjustInputClass:
    def __init__(self, folder_path):
        os.makedirs(os.path.join(folder_path, new_folder), exist_ok=True)
        self.data_ls = self.__get_input(os.path.join(folder_path, "data.txt"))
        self.label_ls = self.__get_input(os.path.join(folder_path, "label.txt"))
        self.count = Counter(self.label_ls)

    def __get_input(self, file_path):
        with open(file_path, "r") as f:
            ls = [int(line[:-1]) for line in f.readlines()]
        f.close()
        return ls

    def get_begin_ls(self, count):
        begin_idx = 0
        begin_ls = [begin_idx]
        for idx in range(len(count)):
            begin_idx += count[idx]
            begin_ls.append(begin_idx)
        return begin_ls


# folder_path = "../5_input/input1"
# os.makedirs(os.path.join(folder_path, new_folder), exist_ok=True)
# data_path = os.path.join(folder_path, "data.txt")
# label_path = os.path.join(folder_path, "label.txt")
#
#
# def get_input(file_path):
#     with open(file_path, "r") as f:
#         ls = [int(line[:-1]) for line in f.readlines()]
#     f.close()
#     return ls
#
#
# def get_begin_ls(count):
#     begin_idx = 0
#     begin_ls = [begin_idx]
#     for idx in range(len(count)):
#         begin_idx += count[idx]
#         begin_ls.append(begin_idx)
#     return begin_ls
#
#
# data_ls = get_input(data_path)
# label_ls = get_input(label_path)
#
# count = Counter(label_ls)
# min_val = min(count.values())




