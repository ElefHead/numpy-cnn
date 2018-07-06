import numpy as np


class Relu:
    def __init__(self):
        self.cache = {}
        self.has_units = False

    def has_weights(self):
        return self.has_units

    def forward_propagate(self, Z, save_cache=False):
        if save_cache:
            self.cache['Z'] = Z
        return np.where(Z >= 0, Z, 0)

    def back_propagate(self, dA):
        Z = self.cache['Z']
        return dA * np.where(Z >= 0, 1, 0)


class Softmax:
    def __init__(self):
        self.cache = {}
        self.has_units = False

    def has_weights(self):
        return self.has_units

    def forward_propagate(self, Z, save_cache=False):
        if save_cache:
            self.cache['Z'] = Z
        Z_ = Z - Z.max()
        e = np.exp(Z_)
        return e / np.sum(e, axis=0, keepdims=True)

    def back_propagate(self, dA):
        Z = self.cache['Z']
        return dA * (Z * (1 - Z))


class Elu:
    def __init__(self, alpha=1.2):
        self.cache = {}
        self.params = {
            'alpha': alpha
        }
        self.has_units = False

    def has_weights(self):
        return self.has_units

    def forward_propagate(self, Z, save_cache=False):
        if save_cache:
            self.cache['Z'] = Z
        return np.where(Z >= 0, Z, self.params['alpha'] * (np.exp(Z) - 1))

    def back_propagate(self, dA):
        alpha = self.params['alpha']
        Z = self.cache['Z']
        return dA * np.where(Z >= 0, 1, self.forward_propagate(Z, alpha) + alpha)


class Selu:
    def __init__(self, alpha=1.6733, selu_lambda=1.0507):
        self.params = {
            'alpha' : alpha,
            'lambda' : selu_lambda
        }
        self.cache = {}
        self.has_units = False

    def has_weights(self):
        return self.has_units

    def forward_propagate(self, Z, save_cache=False):
        if save_cache:
            self.cache['Z'] = Z
        return self.params['lambda'] * np.where(Z >= 0, Z, self.params['alpha'] * (np.exp(Z) - 1))

    def back_propagate(self, dA):
        Z = self.cache['Z']
        selu_lambda, alpha = self.params['lambda'], self.params['alpha']
        return dA * selu_lambda*np.where(Z >= 0, 1, alpha*np.exp(Z))