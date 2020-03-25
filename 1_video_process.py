from src.human_detection import ImgProcessor
import cv2
from config.config import video_process_class, size
import os

IP = ImgProcessor()
store_size = size
dest_folder = "2_frame"


class VideoProcessor:
    def __init__(self, video_path, draw_video_path, output_txt_path):
        self.cap = cv2.VideoCapture(video_path)
        self.height, self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.coord = []
        self.draw_img = draw_video_path

        # self.out = cv2.VideoWriter(draw_video_path, cv2.VideoWriter_fourcc(*'XVID'), 10, store_size)
        self.file = open(output_txt_path, "w")

    def __normalize_coordinates(self, coordinates):
        for i in range(len(coordinates)):
            if (i+1) % 2 == 0:
                coordinates[i] = coordinates[i] / self.height
            else:
                coordinates[i] = coordinates[i] / self.width
        return coordinates

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
                kps, img, black_img = IP.process_img(frame)
                if kps:
                    self.coord = self.__normalize_coordinates(kps)
                    self.__write_txt()

                resize = cv2.resize(img, store_size)
                resize_black = cv2.resize(black_img, store_size)
                cv2.imwrite(os.path.join(self.draw_img, "{}.jpg".format(cnt)), frame)
                # self.out.write(resize)
                cv2.imshow("res", resize_black)
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
        self.draw_video_folder = [path_name.replace("1_video", dest_folder)[:-4] for path_name in self.video_ls]
        self.txt_ls = [(path_name.replace("1_video", "3_coord"))[:-4] + ".txt" for path_name in self.video_ls]
        os.makedirs(os.path.join(dest_folder, folder), exist_ok=True)
        os.makedirs(os.path.join("3_coord", folder), exist_ok=True)
        print("Processing video folder: {}".format(folder))

    def process_folder(self):
        for sv, dv, ot in zip(self.video_ls, self.draw_video_folder, self.txt_ls):
            if os.path.exists(dv):
                print("Video {} has been processed!".format(sv))
                continue

            os.makedirs(dv,exist_ok=True)
            VP = VideoProcessor(sv, dv, ot)
            VP.process_video()
            IP.init_sort()
            print("Finish processing video {}".format(sv))


if __name__ == '__main__':
    for cls in video_process_class:
        VFP = VideoFolderProcessor(cls)
        VFP.process_folder()
