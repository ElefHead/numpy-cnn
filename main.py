from filereader import get_data
from fully_connected import FullyConnected
from convolution import Convolution
from pooling import Pooling
from activation import Relu, Softmax

from utils import get_batches, evaluate

import numpy as np
np.random.seed(0)


NUM_CLASSES = 10
BATCH_SIZE = 256


def predict(model, data):
    predictions = data
    for layer in model:
        print("Type = {}, Input shape = {}".format(type(layer), str(predictions.shape)))
        predictions = layer.forward_propagate(predictions, save_cache=False)
    return predictions


if __name__ == '__main__':
    train_data, train_labels = get_data(num_samples=200)
    test_data, test_labels = get_data(num_samples=100, dataset="testing")

    print("Train data shape: {}, {}".format(train_data.shape, train_labels.shape))
    print("Test data shape: {}, {}".format(test_data.shape, test_labels.shape))

    model = [
        Convolution(filters=12, padding='same'),
        Relu(),
        Pooling(mode='max'),
        FullyConnected(units=10),
        Softmax()
    ]

    predict(model, train_data)