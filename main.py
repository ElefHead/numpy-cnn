from layers.fully_connected import FullyConnected
from layers.convolution import Convolution
from layers.pooling import Pooling
from layers.flatten import Flatten
from layers.activation import Relu, Softmax

from utilities.filereader import get_data
from utilities.model import Model

from loss.losses import CategoricalCrossEntropy

import numpy as np
np.random.seed(0)


if __name__ == '__main__':
    train_data, train_labels = get_data(num_samples=50000)
    test_data, test_labels = get_data(num_samples=10000, dataset="testing")

    print("Train data shape: {}, {}".format(train_data.shape, train_labels.shape))
    print("Test data shape: {}, {}".format(test_data.shape, test_labels.shape))

    model = Model(
        Convolution(filters=12, padding='same'),
        Relu(),
        Pooling(mode='max', kernel_shape=(2, 2), stride=2),
        Flatten(),
        FullyConnected(units=10),
        Softmax()
    )
    model.set_loss(CategoricalCrossEntropy)
    # model.set_batch_size(batch_size=256)

    model.train(train_data, train_labels.T, epochs=5, verbose=True)

    print('Testing accuracy = {}'.format(model.evaluate(test_data, test_labels)))
