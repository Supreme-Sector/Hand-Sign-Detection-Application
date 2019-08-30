# Hand-Sign-Detection-Application
[This repository is still a work in progress] <br>
The code uses a simple deep feed forward neural network to detect a particular hand sign and trigger an event. <br>
A single output neuron will give a 1 if it detects the hand sign and 0 if it doesn't. <br>

## Usage:
Run training_data_collector.py and test_data_collector.py to collect training and test data <br>
Separate Python scripts are needed to load the data using data_loader.py and to create/train a network using network.py. <br>

Example (in Python Shell): <br>
>>> \>>> import data_loader; training_data, test_data=data_loader.load_data(); import network; net=network.Network([19200,500,400,400,1]); net.SGD(training_data,100,50,2.0,test_data=test_data) <br>
