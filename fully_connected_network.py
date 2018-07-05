from filereader import get_data
from fully_connected import FullyConnected
from utils import cross_entropy_loss, d_cross_entropy_loss, get_batches, evaluate

import numpy as np
np.random.seed(0)

NUM_CLASSES = 10
BATCH_SIZE = 256


def train(data, labels, model, batch_size=256, epochs=50,
          optimization='adam'):
    iter = 1
    for epoch in range(epochs):
        for x_batch, y_batch in get_batches(data, labels):
            batch_preds = x_batch.copy()
            for layer in model:
                batch_preds = layer.forward_propagate(batch_preds, save_cache=True)

            dA = d_cross_entropy_loss(y_batch, batch_preds)
            for layer in reversed(model):
                dA = layer.back_propagate(dA, batch_size=x_batch.shape[1])
                if optimization == 'adam':
                    layer.momentum()
                    layer.rmsprop()

            for layer in model:
                layer.apply_grads(optimization=optimization, correct_bias=True, iter=iter)
        predictions = predict(model, data)
        print("Training accuracy (epoch {}): {}".format(epoch+1, evaluate(labels, predictions)))
        print('loss: ', cross_entropy_loss(labels, predictions))
        iter += batch_size
    return model


def predict(model, data):
    predictions = np.zeros(shape=(NUM_CLASSES, data.shape[1]))
    num_batches = data.shape[1] // BATCH_SIZE
    for batch_num, x_batch in enumerate(get_batches(data, shuffle=False)):
        batch_preds = x_batch.copy()
        for layer in model:
            batch_preds = layer.forward_propagate(batch_preds, save_cache=False)
        if batch_num <= num_batches - 1:
            predictions[:, batch_num * BATCH_SIZE:(batch_num + 1) * BATCH_SIZE] = batch_preds
        else:
            predictions[:, batch_num * BATCH_SIZE:] = batch_preds
    return predictions


if __name__ == '__main__':
    train_data, train_labels = get_data()
    test_data, test_labels = get_data(num_samples=10000, dataset="testing")

    train_data = train_data.reshape(-1, 32 * 32 * 3).T / 255
    test_data = test_data.reshape(-1, 32 * 32 * 3).T / 255
    train_labels = train_labels.T
    test_labels = test_labels.T

    print("Train data shape: {}, {}".format(train_data.shape, train_labels.shape))
    print("Test data shape: {}, {}".format(test_data.shape, test_labels.shape))

    model = [
        FullyConnected(units=200, activation='elu'),
        FullyConnected(units=200, activation='elu'),
        FullyConnected(units=10, activation='softmax')
    ]

    model = train(train_data, train_labels, model)
    test_prediction = predict(model, test_data)
    print("Testing accuracy: {}".format(evaluate(test_labels, test_prediction)))
