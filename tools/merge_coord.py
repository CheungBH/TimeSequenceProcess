import os
import shutil

coord_folder = "../3_coord"
target_cls = ["test1", "test2", "test3"]
dest_class = "test"
os.makedirs(os.path.join(coord_folder, dest_class))
src_txt = []

for cls in target_cls:
    assert os.path.isdir(os.path.join(coord_folder, cls)), "Your target class {} is not available".format(cls)
    for txt in os.listdir(os.path.join(coord_folder, cls)):
        src_txt.append(os.path.join(coord_folder, cls, txt))

dest_txt = [os.path.join(coord_folder, dest_class, "{}_{}".format(dest_class, idx)) for idx in range(len(src_txt))]

for s, d in zip(src_txt, dest_txt):
    shutil.copy(s, d)
