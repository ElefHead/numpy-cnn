import numpy as np

from utilities.utils import get_batches, evaluate


class Model:
    def __init__(self, *model):
        self.model = model
        self.num_classes = 0
        self.batch_size = 0
        self.loss = None

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size

    def set_loss(self, loss):
        self.loss = loss

    def train(self, data, labels, batch_size=256, epochs=50, optimization='adam', verbose=False):
        if self.loss is None:
            raise RuntimeError("Set loss first using 'Model.set_loss(<loss>)'")

        self.set_batch_size(batch_size)

        iter = 1
        for epoch in range(epochs):
            for i, (x_batch, y_batch) in enumerate(get_batches(data, labels)):
                batch_preds = x_batch.copy()
                for layer in self.model:
                    batch_preds = layer.forward_propagate(batch_preds, save_cache=True)
                if verbose:
                    print('loss = {}'.format(self.loss.compute_loss(y_batch, batch_preds)))
                    print('batch accuracy (epoch {}, batch {}) = {}'.format(epoch+1, i+1, str(evaluate(y_batch, batch_preds))))
                dA = self.loss.compute_derivative(y_batch, batch_preds)
                for layer in reversed(self.model):
                    print('layer: {}, dA shape: {}'.format(str(type(layer)), str(dA.shape)))
                    dA = layer.back_propagate(dA)
                    if layer.has_weights():
                        if optimization == 'adam':
                            layer.momentum()
                            layer.rmsprop()

                for layer in self.model:
                    if layer.has_weights():
                        layer.apply_grads(optimization=optimization, correct_bias=True, iter=iter)

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
        if (M, N)  == labels.shape:
            return evaluate(labels, predictions)
        elif (N, M) == labels.shape:
            return evaluate(labels, predictions.T)
        else:
            raise RuntimeError("Prediction and label shapes don't match")
