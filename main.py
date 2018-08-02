from layers.fully_connected import FullyConnected
from layers.convolution import Convolution
from layers.pooling import Pooling
from layers.flatten import Flatten
from layers.activation import Elu, Softmax

from utilities.filereader import get_data
from utilities.model import Model
from utilities.utils import evaluate

from loss.losses import CategoricalCrossEntropy

import numpy as np
np.random.seed(0)


if __name__ == '__main__':
    train_data, train_labels = get_data(num_samples=500)
    test_data, test_labels = get_data(num_samples=500, dataset="testing")

    train_data = train_data / 255
    test_data = test_data / 255

    print("Train data shape: {}, {}".format(train_data.shape, train_labels.shape))
    print("Test data shape: {}, {}".format(test_data.shape, test_labels.shape))

    model = Model(
        Convolution(filters=5, padding='same'),
        Elu(),
        Pooling(mode='max', kernel_shape=(2, 2), stride=2),
        Flatten(),
        FullyConnected(units=10),
        Softmax(),
        name='cnn5'
    )

    model.set_loss(CategoricalCrossEntropy)

    model.train(train_data, train_labels.T, epochs=5)
    prediction = model.predict(test_data)

    print('Testing accuracy = {}'.format(evaluate(test_labels.T, prediction)))
