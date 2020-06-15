import cv2
import os

src_folder = "../7_test/test_v/video"
video_names = [vn for vn in os.listdir(src_folder)]
dest_folder = src_folder + "_cnt"
os.makedirs(dest_folder)

for video in video_names:
    cap = cv2.VideoCapture(os.path.join(src_folder, video))
    cnt_video = os.path.join(dest_folder, video)
    (h, w) = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    out = cv2.VideoWriter(cnt_video, cv2.VideoWriter_fourcc(*'XVID'), 10, (w, h))
    cnt = 0
    while True:
        cnt += 1
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (w, h))
            cv2.putText(frame, "cnt: {}".format(cnt), (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 3)
            out.write(frame)
        else:
            break

