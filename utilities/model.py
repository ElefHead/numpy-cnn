import numpy as np

from utilities.utils import get_batches, evaluate


class Model:
    def __init__(self, *model):
        self.model = model
        self.num_classes = 0
        self.batch_size = 0

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size

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