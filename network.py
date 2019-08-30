# This code is based off of mnielsen's work with the following modifications:
#   1. Python 3 compatibility
#   2. Exponential Decay of the learning rate
#   3. Implements a fully matrix based approach for each mini batch
# The original code can be found at https://github.com/mnielsen/neural-networks-and-deep-learning/blob/master/src/network.py

"""
network.py
~~~~~~~~~~

A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network.  Gradients are calculated
using backpropagation.  Note that I have focused on making the code
simple, easily readable, and easily modifiable.  It is not optimized,
and omits many desirable features.
"""

#### Libraries
import random
import numpy as np
import _pickle as cPickle

class Network:

    def __init__(self, sizes):
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers."""
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) * 4 for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) * 4 for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""


        if test_data:
            n_test = len(test_data)
        n = len(training_data)
        max_accuracy=0

        for j in range(epochs):
            #Exponential decay of step size
            new_eta=eta*np.exp(-0.03*j)

            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, new_eta)
            if test_data:
                number_right=self.evaluate(test_data)
                percent_accuracy=round(number_right/n_test*100, 1)
                if percent_accuracy > max_accuracy:
                    max_accuracy=percent_accuracy
                    pickle_out=open("neural_network.pickle","wb")
                    cPickle.dump(self, pickle_out)
                    pickle_out.close()

                print("Epoch {}: {} / {} = {}% accuracy".format(
                    j, number_right, n_test, percent_accuracy))
            else:
                print("Epoch {} complete".format(j))

        print("Maximum accuracy is {}%".format(max_accuracy))

    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        x=np.column_stack([example[0] for example in mini_batch])
        y=np.column_stack([example[1] for example in mini_batch])

        nabla_b, nabla_w = self.backprop(x, y)



        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]

        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the sum of the
        gradient vectors of the cost function among the training examples in a mini batch.
        ``nabla_b`` and ``nabla_w`` are layer-by-layer lists of numpy arrays,
        similar to ``self.biases`` and ``self.weights``."""

        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)


        # backward pass
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])

        nabla_b[-1]=delta.sum(axis=1).reshape((len(delta),1))
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l]=delta.sum(axis=1).reshape((len(delta),1))
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result."""
        test_results = [(np.round(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)

#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))
