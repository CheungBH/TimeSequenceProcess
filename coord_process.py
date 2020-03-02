import config.config as c
import os

step = c.coord_step
frame = c.coord_frame
method = c.coord_process_method
cls_ls = c.coord_process_class


class CoordFolderProcessor:
    def __init__(self, cls, frame, step):
        self.frame, self.step, self.coord_folder = frame, step, cls
        self.coord_folder = os.path.join("3_coord", cls)
        self.dest_folder = os.path.join("4_data", method, "{}frames".format(frame), "{}steps".format(step))
        os.makedirs(os.path.join(self.dest_folder, cls))
        self.coord_ls = [os.path.join(self.coord_folder, txt_name) for txt_name in os.listdir(self.coord_folder)]
        self.dest_ls = [path_name.replace("3_coord", self.dest_folder) for path_name in self.coord_ls]

    def process_folder(self):
        for idx in range(len(self.coord_ls)):
            self.process(self.coord_ls[idx], self.dest_ls[idx])

    def process(self, coord_path, dest_path):
        with open(coord_path, "r") as rf:
            lines = [line for line in rf.readlines()]
            rf.close()

        outs = []
        try:
            for begin_idx in range(len(lines))[::self.step]:
                out = ""
                for idx in range(self.frame):
                    out += lines[begin_idx + idx].replace("\n", "\t")
                outs.append(out[:-1] + '\n')
        except IndexError:
            pass

        with open(dest_path, "w") as wf:
            for out in outs:
                wf.write(out)


if __name__ == '__main__':
    for cls in cls_ls:
        assert os.path.exists("3_coord/{}".format(cls)), "The coordinate folder doesn't exist! Please run ‘video_process.py’ to generate the coordinate files first"
        CFP = CoordFolderProcessor(cls, frame, step)
        CFP.process_folder()
