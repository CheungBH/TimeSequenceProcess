import random
import os


class txtGenerator:
    def __init__(self, cls_name):
        self.folder_name = cls_name
        self.folder_path = "../3_coord/{}".format(cls_name)
        os.makedirs(self.folder_path)
        self.files_num = 45
        self.basic_line = 60

    def generate_file(self, line_num, file_num):
        file = open("{}/{}_{}.txt".format(self.folder_path, self.folder_name, file_num), "w")
        lines = [[i for i in range(j,34+j)] for j in range(line_num)]
        for line in lines:
            write_line = ""
            for idx in range(len(line)):
                write_line += str(line[idx])
                if idx == 33:
                    write_line += "\n"
                else:
                    write_line += "\t"
            file.write(write_line)
        file.close()

    def generate(self):
        for idx in range(self.files_num):
            self.generate_file(self.basic_line + random.randint(-10,10), idx)


if __name__ == '__main__':
    for cls in ["normal", "drown"]:
        txtG = txtGenerator(cls)
        txtG.generate()
