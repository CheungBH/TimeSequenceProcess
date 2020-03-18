import cv2


def run_multi_cam(cam_ls):
	fourcc = cv2.VideoWriter_fourcc(*'XVID')
	caps = [cv2.VideoCapture(cam) for cam in cam_ls]
	height, width = int(caps[0].get(3)), int(caps[0].get(4))
	outs = [cv2.VideoWriter('video{}.avi'.format(cam), fourcc, 10, (height, width)) for cam in cam_ls]

	while True:
		try:
			res = [(cap.read()) for cap in caps]
			if res[0]:
				for idx in range(len(cam_ls)):
					outs[idx].write(res[idx][1])
					cv2.imshow("res{}".format(idx), res[idx][1])
				cv2.waitKey(2)
		except KeyboardInterrupt:
			cv2.destroyAllWindows()
			break


if __name__ == '__main__':
	run_multi_cam([0,1,2])
