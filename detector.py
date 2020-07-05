import numpy as np

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
