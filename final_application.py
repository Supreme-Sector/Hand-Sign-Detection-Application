import _pickle as cPickle
import network
import numpy as np
import cv2
import time
from detector import Detector
import os

def vectorize_frame_data(frame):
    input_vector=frame.flatten()
    input_vector=input_vector.reshape(len(input_vector),1)
    return input_vector

def open_youtube():
    os.system("start https://youtube.com")

pickle_in = open("final_detector.pickle", "rb")
detector = cPickle.load(pickle_in)
pickle_in.close()

cap = cv2.VideoCapture(0)
detector_history = [0] * 20
i = 0

while True:
    ret, frame = cap.read()
    simple_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    simple_frame=cv2.resize(simple_frame,(int(frame.shape[1]*0.125),int(frame.shape[0]*0.125)))
    cv2.imshow("Y for YouTube", frame)
    cv2.waitKey(1)
    result = detector.detect(vectorize_frame_data(simple_frame))
    print(result)
    detector_history[i] = result
    i = (i+1) % 20
    if(sum(detector_history)>=12):
        open_youtube()
        time.sleep(3)
        detector_history = [0] * 20
        i = 0
