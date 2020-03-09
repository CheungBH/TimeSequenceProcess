import os
from collections import Counter
import random
import shutil

err = 0.05
new_folder = "equal"


class AdjustInputClass:
    def __init__(self, folder_path):
        self.folder = folder_path
        os.makedirs(os.path.join(folder_path, new_folder), exist_ok=True)
        self.data_ls = self.__get_input(os.path.join(folder_path, "data.txt"))
        self.label_ls = self.__get_input(os.path.join(folder_path, "label.txt"))
        self.count = Counter(self.label_ls)
        self.min_val = min(self.count.values())
        self.boundary = [0]

    def __get_input(self, file_path):
        with open(file_path, "r") as f:
            ls = [line for line in f.readlines()]
        return ls

    def __get_boundary(self):
        n = 0
        for k, v in self.count.items():
            n += v
            self.boundary.append(n)

    def __get_random_idx(self):
        select = []
        for idx, (k, v) in enumerate(self.count.items()):
            if v != self.min_val:
                nums = random.sample(range(self.boundary[0], self.boundary[1]), v - self.min_val)
                select += [n for n in nums]
        return select

    def adjust_sample(self):
        self.__get_boundary()
        self.boundary = [0, 1226, 1731]
        select = self.__get_random_idx()

        new_data = [line for idx, line in enumerate(self.data_ls) if idx not in select]
        new_label = [item for idx, item in enumerate(self.label_ls) if idx not in select]
        with open(os.path.join(self.folder, new_folder, "data.txt"), "w") as data_f:
            for line in new_data:
                data_f.write(line)

        with open(os.path.join(self.folder, new_folder, "label.txt"), "w") as label_f:
            for line in new_label:
                label_f.write(line)

        shutil.copy(os.path.join(self.folder, "cls.txt"), os.path.join(self.folder, new_folder, "cls.txt"))


if __name__ == '__main__':
    AIC = AdjustInputClass("../5_input/input1")
    AIC.adjust_sample()



# folder_path = "../tmp/test"
# os.makedirs(os.path.join(folder_path, new_folder), exist_ok=True)
# data_path = os.path.join(folder_path, "data.txt")
# label_path = os.path.join(folder_path, "label.txt")
#
#
# def get_input(file_path):
#     with open(file_path, "r") as f:
#         ls = [line for line in f.readlines()]
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


# data_ls = get_input(data_path)
# label_ls = get_input(label_path)
#
# count = Counter(label_ls)
# min_val = min(count.values())
# # min_item = count.mo
#
# boundary, num = [0], 0
# for k, v in count.items():
#     num += v
#     boundary.append(num)
#
# select = []
# for idx, (k,v) in enumerate(count.items()):
#     if v != min_val:
#         nums = random.sample(range(boundary[0], boundary[1]), v - min_val)
#         select += [n for n in nums]
#         nums = []
#
# new_data = [line for idx, line in enumerate(data_ls) if idx not in select]
# new_label = [item for idx, item in enumerate(label_ls) if idx not in select]
#
# a = 1


