import cv2

video_path = "1_video/test/others/6c_Trim1.mp4"

cap = cv2.VideoCapture(video_path)

while True:
    ret, frame = cap.read()
    if ret:
        frame = cv2.resize(frame, (540, 360))
        cv2.imshow("origin", frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow("gray", gray)
        cv2.waitKey(10)
    else:
        break
cv2.destroyAllWindows()

