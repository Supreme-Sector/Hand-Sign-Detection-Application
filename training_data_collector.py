import numpy as np
import cv2
import _pickle as cPickle

def vectorize_frame_data(frame):
    input_vector=frame.flatten()
    input_vector=input_vector.reshape(len(input_vector),1)
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
result=np.array([[int(input("Enter 1 for POSITIVE data or 0 for NEGATIVE data: "))]])

if result != 1 and result != 0:
    print("Invalid number")
    exit()

cap = cv2.VideoCapture(0)

for i in range(5000):
    ret, frame = cap.read()
    simple_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    simple_frame=cv2.resize(simple_frame,(int(frame.shape[1]*0.25),int(frame.shape[0]*0.25)))
    cv2.imshow("Gathering image data...", frame)
    data.append((vectorize_frame_data(simple_frame),result))
    c=cv2.waitKey(1)
    if c==ord('q'):
        stop()
    elif c==ord('k'):
        pause()

print("Number of examples: {}".format(len(data)))

if result==1:
    pickle_out=open("./data/training_data/POSITIVE_data/positive_data.pickle", "wb")
elif result==0:
    pickle_out=open("./data/training_data/NEGATIVE_data/negative_data.pickle", "wb")

cPickle.dump(data, pickle_out)
pickle_out.close()
exit()
