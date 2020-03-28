import os
from collections import Counter, OrderedDict
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
        self.count = OrderedDict(Counter(self.label_ls))
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
        # self.boundary = self.__get_boundary()
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
    AIC = AdjustInputClass("../5_input/input2")
    AIC.adjust_sample()


