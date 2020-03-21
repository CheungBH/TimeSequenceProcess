import config.config as c
import os
import numpy as np
from utils.utils import numpy2str

step = c.coord_step
frame = c.coord_frame
method = c.coord_process_method
cls_ls = c.coord_process_class


class CoordFolderProcessor:
    def __init__(self, clss, frm, stp):
        self.frame, self.step, self.coord_folder = frm, stp, clss
        self.coord_folder = os.path.join("3_coord", cls)
        self.dest_folder = os.path.join("4_data", method, "{}frames".format(frame), "{}steps".format(step))
        os.makedirs(os.path.join(self.dest_folder, cls))
        self.coord_ls = [os.path.join(self.coord_folder, txt_name) for txt_name in os.listdir(self.coord_folder)]
        self.dest_ls = [path_name.replace("3_coord", self.dest_folder) for path_name in self.coord_ls]

    def process_folder(self):
        for idx in range(len(self.coord_ls)):
            self.__process(self.coord_ls[idx], self.dest_ls[idx])

    def __process(self, coord_path, dest_path):
        lines = self.process_line(coord_path)

        outs = []
        try:
            for begin_idx in range(len(lines))[::self.step]:
                out = ""
                for idx in range(self.frame):
                    out += numpy2str(lines[begin_idx + idx])
                    # out += lines[begin_idx + idx]
                outs.append(out[:-1] + '\n')
        except IndexError:
            pass

        with open(dest_path, "w") as wf:
            for out in outs:
                wf.write(out)

    def process_line(self, file):
        return np.loadtxt(file)


class CoordinateFolderProcessorReducePoint(CoordFolderProcessor):
    def __init__(self, clss, frm, stp):
        super().__init__(clss, frm, stp)
        self.selected_point = [1,2,5,6,7,8,9,10,11]

    def process_line(self, file):
        lines = []
        txt = np.loadtxt(file)
        for t in txt:
            idx = [2*(num-1) for num in self.selected_point] + [2*(num-1)+1 for num in self.selected_point]
            lines.append([t[i] for i in idx])
        return lines


if __name__ == '__main__':
    for cls in cls_ls:
        assert os.path.exists("3_coord/{}".format(cls)), "The coordinate folder doesn't exist! Please run " \
                                                         "‘video_process.py’ to generate the coordinate files first"
        if method == "ordinary":
            CFP = CoordFolderProcessor(cls, frame, step)
            CFP.process_folder()
        elif "point" in method:
            CFPRP = CoordinateFolderProcessorReducePoint(cls, frame, step)
            CFPRP.process_folder()

