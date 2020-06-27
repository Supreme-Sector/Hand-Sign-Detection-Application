import data_loader
import network

training_data, test_data = data_loader.load_data()
net = network.Network([4800, 500, 30, 1]) # Try to experiment with different layer sizes
net.SGD(training_data, 80, 20, 2.0, test_data=test_data) # Try to experiment with these training hyperparameters
