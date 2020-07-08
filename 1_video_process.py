from src.human_detection import HumanDetection
import cv2
from config.config import video_process_class, size, save_frame, save_black_img, save_kps_img, save_kps_video, process_gray
import os
from utils.kp_process import KPSProcessor

IP = HumanDetection()
store_size = size
dest_folder = "2_kps_video"


class VideoProcessor:
    def __init__(self, video_path, draw_video_path, output_txt_path):
        self.cap = cv2.VideoCapture(video_path)
        self.coord = []
        # self.draw_img = draw_video_path
        self.out = cv2.VideoWriter(draw_video_path, cv2.VideoWriter_fourcc(*'XVID'), 10, store_size)
        self.file = open(output_txt_path, "w")
        self.KPSP = KPSProcessor(int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)))

    def __write_txt(self):
        for item in self.coord:
            self.file.write(str(item)+"\t")
        self.file.write("\n")

    def process_video(self):
        cnt = 0
        while True:
            cnt += 1
            # print("Current frame is {}".format(cnt))
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.resize(frame, store_size)
                kps, box, ks = IP.process_img(frame, gray=process_gray)
                img, black_img = IP.visualize()

                if kps:
                    self.coord = self.KPSP.process_single_kps(kps, 1)
                    self.__write_txt()

                resize = cv2.resize(img, store_size)
                # resize_black = cv2.resize(black_img, store_size)
                # cv2.imwrite(os.path.join(self.draw_img, "{}.jpg".format(cnt)), frame)
                self.out.write(resize)
                cv2.imshow("res", resize)
                cv2.waitKey(2)

            else:
                self.cap.release()
                self.file.close()
                cv2.destroyAllWindows()
                # self.out.release()
                break


class VideoFolderProcessor:
    def __init__(self, folder):
        self.video_ls = [os.path.join("1_video", folder, video_name) for video_name in os.listdir(os.path.join("1_video", folder))]
        self.kps_videos = [path_name.replace("1_video", dest_folder)[:-4]+".avi" for path_name in self.video_ls]
        self.txt_ls = [(path_name.replace("1_video", "3_coord"))[:-4] + ".txt" for path_name in self.video_ls]
        os.makedirs(os.path.join(dest_folder, folder), exist_ok=True)
        os.makedirs(os.path.join("3_coord", folder), exist_ok=True)
        print("Processing video folder: {}".format(folder))

    def process_folder(self):
        for sv, kv, ot in zip(self.video_ls, self.kps_videos, self.txt_ls):
            IP.init_sort()
            if os.path.exists(kv):
                print("Video {} has been processed!".format(sv))
                continue

            # os.makedirs(dv,exist_ok=True)
            VP = VideoProcessor(sv, kv, ot)
            VP.process_video()

            print("Finish processing video {}".format(sv))


if __name__ == '__main__':
    for cls in video_process_class:
        VFP = VideoFolderProcessor(cls)
        IP.init_sort()
        VFP.process_folder()
