import numpy as np

from os import makedirs, path

from utilities.utils import get_batches, evaluate
from utilities.settings import set_network_name, get_models_path

class Model:
    def __init__(self, *model, **kwargs):
        self.model = model
        self.num_classes = 0
        self.batch_size = 0
        self.loss = None
        self.optimizer = None
        self.name = kwargs['name'] if 'name' in kwargs else None

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size

    def set_loss(self, loss):
        self.loss = loss

    def set_name(self, name):
        set_network_name(name)

    def load_weights(self):
        for layer in self.model:
            if layer.has_weights():
                layer.load_weights(path.join(get_models_path(), self.name))

    def train(self, data, labels, batch_size=256, epochs=50, optimization='adam',
              save_model=True, load_and_continue=False):
        if self.loss is None:
            raise RuntimeError("Set loss first using 'model.set_loss(<loss>)'")

        self.set_batch_size(batch_size)
        if save_model:
            self.set_name(self.name)

        if load_and_continue:
            for layer in self.model:
                if layer.has_weights():
                    layer.load_weights(path.join(get_models_path(), self.name))

        iter = 1
        for epoch in range(epochs):
            print('Running Epoch:', epoch + 1)
            for i, (x_batch, y_batch) in enumerate(get_batches(data, labels)):
                batch_preds = x_batch.copy()
                for num, layer in enumerate(self.model):
                    batch_preds = layer.forward_propagate(batch_preds, save_cache=True)
                dA = self.loss.compute_derivative(y_batch, batch_preds)
                for layer in reversed(self.model):
                    dA = layer.back_propagate(dA)
                    if layer.has_weights():
                        if optimization == 'adam':
                            layer.momentum()
                            layer.rmsprop()

                for layer in self.model:
                    if layer.has_weights():
                        layer.apply_grads(optimization=optimization, correct_bias=True, iter=iter)
            for layer in self.model:
                if layer.has_weights():
                    layer.save_weights(path.join(get_models_path(), self.name))

            iter += batch_size

    def predict(self, data):
        if self.batch_size == 0:
            self.batch_size = data.shape[0]
        if self.num_classes == 0:
            predictions = np.zeros((1, data.shape[0]))
        else:
            predictions = np.zeros((self.num_classes, data.shape[0]))
        num_batches = data.shape[0] // self.batch_size
        for batch_num, x_batch in enumerate(get_batches(data, batch_size=self.batch_size, shuffle=False)):
            batch_preds = x_batch.copy()
            for layer in self.model:
                batch_preds = layer.forward_propagate(batch_preds, save_cache=False)
            M, N = batch_preds.shape
            if M != predictions.shape[0]:
                predictions = np.zeros(shape=(M, data.shape[0]))
            if batch_num <= num_batches - 1:
                predictions[:, batch_num * self.batch_size:(batch_num + 1) * self.batch_size] = batch_preds
            else:
                predictions[:, batch_num * self.batch_size:] = batch_preds
        return predictions

    def evaluate(self, data, labels):
        predictions = self.predict(data)
        M, N = predictions.shape
        if (M, N) == labels.shape:
            return evaluate(labels, predictions)
        elif (N, M) == labels.shape:
            return evaluate(labels.T, predictions)
        else:
            raise RuntimeError("Prediction and label shapes don't match")
