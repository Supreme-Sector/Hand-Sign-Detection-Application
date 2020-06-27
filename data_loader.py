import _pickle as cPickle

def load_data():
    pickle_in=open("./data/training_data/NEGATIVE_data/negative_data.pickle","rb")
    examples_negative=cPickle.load(pickle_in)
    pickle_in.close()
    pickle_in=open("./data/training_data/POSITIVE_data/positive_data.pickle","rb")
    examples_positive=cPickle.load(pickle_in)
    pickle_in.close()
    training_data = examples_negative + examples_positive
    pickle_in=open("./data/test_data/test_data.pickle","rb")
    test_data=cPickle.load(pickle_in)
    pickle_in.close()
    return (training_data, test_data)
