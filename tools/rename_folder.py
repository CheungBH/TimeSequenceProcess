import os

folder = "../1_video/drown"

paths = []
for root, dirs, files in os.walk(folder):
    path = [os.path.join(root, name) for name in files]
    paths.extend(path)

for idx, img_path in enumerate(paths):
    os.rename(img_path, os.path.join(folder, "video_drown_{}.mp4".format(idx)))

