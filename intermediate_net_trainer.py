import network
import _pickle as cPickle

pickle_in = open("./data/ensemble_data/ensemble_training_data.pickle", "rb")
training_data = cPickle.load(pickle_in)
pickle_in.close()

net = network.Network([5, 1]) # 5 input neurons for 5 subnets
net.SGD(training_data, 10, 30, 1.0, test_data=training_data) # Try to experiment with different hyperparameters
