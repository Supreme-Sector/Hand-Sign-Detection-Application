"""
How to use:
Run script
Press enter to continue
Feed positive images through webcam
After that, press enter again
Feed negative images thorugh webcam
"""

import numpy as np
import cv2
import _pickle as cPickle

def vectorize_frame_data(frame):
    input_vector=frame.flatten()
    input_vector=input_vector.reshape(len(input_vector),1)/255
    return input_vector

def stop():
    cap.release()
    cv2.destroyAllWindows()
    exit()

def pause():
    while True:
        c=cv2.waitKey(1)
        if c==ord('k'):
            break
        if c==ord('q'):
            stop()

data=[]

result=np.array([[1]])
input("Prepare to feed positive data. Press ENTER to continue...")

cap = cv2.VideoCapture(0)

for i in range(1500):
    ret, frame = cap.read()
    simple_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    simple_frame=cv2.resize(simple_frame,(int(frame.shape[1]*0.125),int(frame.shape[0]*0.125)))
    cv2.imshow("Gathering image data...", frame)
    data.append((vectorize_frame_data(simple_frame),result))
    c=cv2.waitKey(1)
    if c==ord('q'):
        stop()
    elif c==ord('k'):
        pause()

cv2.destroyAllWindows()
print("Positive data gathering complete.")

result=np.array([[0]])
input("Prepare to feed negative data. Press ENTER to continue...")

for i in range(1500):
    ret, frame = cap.read()
    simple_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    simple_frame=cv2.resize(simple_frame,(int(frame.shape[1]*0.125),int(frame.shape[0]*0.125)))
    cv2.imshow("Gathering image data...", frame)
    data.append((vectorize_frame_data(simple_frame),result))
    c=cv2.waitKey(1)
    if c==ord('q'):
        stop()
    elif c==ord('k'):
        pause()

cv2.destroyAllWindows()
print("Negative data gathering complete.")

cap.release()
print("Number of examples: {}".format(len(data)))
pickle_out=open("./data/test_data/test_data.pickle", "wb")
cPickle.dump(data, pickle_out)
pickle_out.close()
exit()
