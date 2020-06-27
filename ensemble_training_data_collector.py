import network
import numpy as np
import cv2
import _pickle as cPickle

files=["1_neural_network_87_5.pickle", "2_neural_network_82_3.pickle", "3_neural_network_80_3.pickle", "4_neural_network_83_1.pickle", "5_neural_network_87_7.pickle"]
subnets=[]


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

for file in files:
    pickle_in=open("./network_pickles/"+file, "rb")
    net=cPickle.load(pickle_in)
    subnets.append(net)

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

training_data = []

for (x, y) in data:
    outputs=[]
    for subnet in subnets:
        outputs.append(subnet.feedforward(x)[0][0])
    outputs=np.asarray(outputs)
    outputs=outputs.reshape(len(outputs), 1)
    training_data.append((outputs, y))

pickle_out=open("./data/ensemble_data/ensemble_training_data.pickle", "wb")
cPickle.dump(training_data, pickle_out)
pickle_out.close()
