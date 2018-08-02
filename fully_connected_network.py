from layers.fully_connected import FullyConnected
from layers.flatten import Flatten
from layers.activation import Elu, Softmax

from utilities.filereader import get_data
from utilities.model import Model

from loss.losses import CategoricalCrossEntropy


import numpy as np
np.random.seed(0)


if __name__ == '__main__':
    train_data, train_labels = get_data()
    test_data, test_labels = get_data(num_samples=10000, dataset="testing")

    train_data = train_data / 255
    test_data = test_data / 255
    train_labels = train_labels.T
    test_labels = test_labels.T

    print("Train data shape: {}, {}".format(train_data.shape, train_labels.shape))
    print("Test data shape: {}, {}".format(test_data.shape, test_labels.shape))

    model = Model(
        Flatten(),
        FullyConnected(units=200),
        Elu(),
        FullyConnected(units=200),
        Elu(),
        FullyConnected(units=10),
        Softmax(),
        name='fcn200'
    )

    model.set_loss(CategoricalCrossEntropy)
    model.train(train_data, train_labels, batch_size=128, epochs=50)

    print('Testing accuracy = {}'.format(model.evaluate(test_data, test_labels)))

