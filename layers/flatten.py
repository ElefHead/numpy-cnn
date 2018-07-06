import numpy as np


class Flatten:
    def __init__(self, transpose=True):
        self.shape = ()
        self.transpose = transpose
        self.has_units = False

    def has_weights(self):
        return self.has_units

    def forward_propagate(self, Z, save_cache=False):
        shape = Z.shape
        if save_cache:
            self.shape = shape
        data = np.ravel(Z).reshape(shape[0], -1)
        if self.transpose:
            data = data.T
        return data

    def back_propagate(self, Z):
        if self.transpose:
            Z = Z.T
        return Z.reshape(self.shape)
