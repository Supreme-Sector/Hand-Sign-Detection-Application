import data_loader
import network
import _pickle as cPickle

f=open("neural_network.pickle","rb")
net=cPickle.load(f)
f.close()

training_data, test_data=data_loader.load_data()

num_correct = net.evaluate(training_data)

print("{}/{} correct".format(num_correct, len(training_data)))
