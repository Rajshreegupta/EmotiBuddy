import cv2
from model import FacialExpressionModel
import numpy as np

facec = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
model = FacialExpressionModel("model.json", "model_weights.h5")
cam = cv2.VideoCapture(0)

cv2.namedWindow("test")

img_counter = 0

while True:
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        break
    # cv2.imshow("test", frame)

    k = cv2.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 13:
        # SPACE pressed
        # img_name = "opencv_frame_{}.png".format(img_counter)
        # cv2.imwrite(img_name, frame)
        # print("{} written!".format(img_name))

        gray_fr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = facec.detectMultiScale(gray_fr, 1.3, 5)
        # print(faces)
        for (x, y, w, h) in faces:
            fc = gray_fr[y:y + h, x:x + w]

            roi = cv2.resize(fc, (48, 48))
            # cv2.imshow('img', fc)
            pred = model.predict_emotion(roi[np.newaxis, :, :, np.newaxis])
        print(pred)
        # print("end")

        # roi=cv2.resize(face,(48,48))
        # pred = model.predict_emotion(roi[np.newaxis, :, :, np.newaxis])
        # print(pred)
        img_counter += 1

cam.release()

cv2.destroyAllWindows()