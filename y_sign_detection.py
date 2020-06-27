import network
import data_loader
import numpy as np
import _pickle as cPickle

class Detector:

    def __init__(self, subnets, merger):
        self.subnets = subnets
        self.merger = merger

    def detect(self, img):
        outputs=[]
        for subnet in self.subnets:
            outputs.append(subnet.feedforward(img)[0][0])
        outputs=np.asarray(outputs)
        outputs=outputs.reshape(len(outputs),1)
        result = self.merger.feedforward(outputs)[0][0]
        result = int(result+0.5)
        return result

    def evaluate(self, test_data):
        test_results = [(self.detect(x), y[0][0]) for (x, y) in test_data]
        return sum((x==y) for (x, y) in test_results)


files=["1_neural_network_87_5.pickle", "2_neural_network_82_3.pickle", "3_neural_network_80_3.pickle", "4_neural_network_83_1.pickle", "5_neural_network_87_7.pickle"]
subnets=[]

for file in files:
    pickle_in=open("./network_pickles/"+file, "rb")
    net=cPickle.load(pickle_in)
    subnets.append(net)

pickle_in=open("./network_pickles/merger_neural_network.pickle", "rb")
merger=cPickle.load(pickle_in)

detector = Detector(subnets, merger)

___, test_data = data_loader.load_data()
n_test = len(test_data)
num_correct = detector.evaluate(test_data)
percent_accuracy=round(num_correct/n_test*100, 1)
print("{}/{} correct = {}% accuracy".format(num_correct, n_test, percent_accuracy))
