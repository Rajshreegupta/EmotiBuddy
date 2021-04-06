# Program to accept client request
# Author @inforkgodara

import socket
import cv2
from model import FacialExpressionModel
import numpy as np

facec = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
model = FacialExpressionModel("model.json", "model_weights.h5")
cam = cv2.VideoCapture(0)

cv2.namedWindow("test")

img_counter = 0

s = socket.socket()
host = socket.gethostname()
print(' Server will start on host : ', host)
port = 8080
s.bind((host, port))
print()
print('Waiting for connection')
print()
s.listen(1)
conn, addr = s.accept()
print(addr, ' Has connected to the server')
print()
while 1:

    message = input(str('>> '))
    message = message.encode()
    conn.send(message)
    ret, frame = cam.read()
    print('Sent')
    print()


    gray_fr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = facec.detectMultiScale(gray_fr, 1.3, 5)
    for (x, y, w, h) in faces:
        fc = gray_fr[y:y + h, x:x + w]

        roi = cv2.resize(fc, (48, 48))
        pred = model.predict_emotion(roi[np.newaxis, :, :, np.newaxis])
    msg=pred

    msg = msg.encode()
    conn.send(msg)
    incoming_message = conn.recv(1024)
    incoming_message = incoming_message.decode()
    print(' Client : ', incoming_message)
    print()
    incoming_message = conn.recv(1024)
    incoming_message = incoming_message.decode()
    print(' Client is ', incoming_message)