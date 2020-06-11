import config.config as c
import os

step, frame, method = c.merge_step, c.merge_frame, c.merge_process_method
src_folder = os.path.join("4_data", method, "{}frames".format(frame), "{}steps".format(step))
assert os.path.isdir(src_folder), "The source folder'name is wrong"
dest_name = c.merge_dest_name
merge_class = c.merge_class
comment = c.merge_comment


class InputMerger:
    def __init__(self):
        self.description = open("5_input/description.txt", "a+")
        self.src_class_ls = [os.path.join(src_folder, cls) for cls in merge_class]
        self.src_label = list(range(len(self.src_class_ls)))
        self.dest_folder = os.path.join("5_input", dest_name)
        os.makedirs(self.dest_folder)
        self.data = open(os.path.join(self.dest_folder, "data.txt"), "w")
        self.label = open(os.path.join(self.dest_folder, "label.txt"), "w")
        with open(os.path.join(self.dest_folder, "cls.txt"), "w") as f:
            for cls in merge_class:
                f.write(cls)
                f.write('\n')


    def __merge_cls_input(self, label, data):
        for out in data:
            self.data.write(out)
            self.label.write(str(label)+'\n')

    def merge(self):
        for idx, src in enumerate(self.src_class_ls):
            txt_ls = [os.path.join(src, txt_name) for txt_name in os.listdir(src)]
            out = []
            for txt in txt_ls:
                with open(txt, "r") as f:
                    for line in f.readlines():
                        out.append(line)
            self.__merge_cls_input(idx, out)
        self.description.write(comment + '\n')
        self.data.close()
        self.label.close()
        self.description.close()


if __name__ == '__main__':
    IM = InputMerger()
    IM.merge()
