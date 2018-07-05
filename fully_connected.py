import numpy as np
from activation import ACTIVATE as activate, D_ACTIVATE as d_activate
from initializers import he_normal

np.random.seed(0)


class FullyConnected:
    def __init__(self, units=200, activation='elu'):
        self.params = {
            'units': units,
            'activation': activate[activation],
            'd_activation': d_activate[activation]
        }
        self.cache = {}
        self.grads = {}
        self.momentum_cache = {}
        self.rmsprop_cache = {}

    def forward_propagate(self, X, save_cache=False):
        if 'W' not in self.params:
            self.params['W'], self.params['b'] = he_normal((X.shape[0], self.params['units']))
        Z = np.dot(self.params['W'], X) + self.params['b']
        if save_cache:
            self.cache['A'] = X
            self.cache['Z'] = Z
        return self.params['activation'](Z)

    def back_propagate(self, dA, batch_size=256):
        dZ = dA * self.params['d_activation'](self.cache['Z'])
        self.grads['dW'] = np.dot(dZ, self.cache['A'].T) / batch_size
        self.grads['db'] = np.sum(dZ, axis=1, keepdims=True)
        return np.dot(self.params['W'].T, dZ)

    def init_cache(self):
        cache = dict()
        cache['dW'] = np.zeros_like(self.params['W'])
        cache['db'] = np.zeros_like(self.params['b'])
        return cache

    def momentum(self, beta=0.9):
        if not self.momentum_cache:
            self.momentum_cache = self.init_cache()
        self.momentum_cache['dW'] = beta * self.momentum_cache['dW'] + (1 - beta) * self.grads['dW']
        self.momentum_cache['db'] = beta * self.momentum_cache['db'] + (1 - beta) * self.grads['db']

    def rmsprop(self, beta=0.999, amsprop=True):
        if not self.rmsprop_cache:
            self.rmsprop_cache = self.init_cache()

        new_dW = beta * self.rmsprop_cache['dW'] + (1 - beta) * (self.grads['dW']**2)
        new_db = beta * self.rmsprop_cache['db'] + (1 - beta) * (self.grads['db']**2)

        if amsprop:
            self.rmsprop_cache['dW'] = np.maximum(self.rmsprop_cache['dW'], new_dW)
            self.rmsprop_cache['db'] = np.maximum(self.rmsprop_cache['db'], new_db)
        else:
            self.rmsprop_cache['dW'] = new_dW
            self.rmsprop_cache['db'] = new_db

    def apply_grads(self, learning_rate=0.001, l2_penalty=1e-4, optimization='adam', epsilon=1e-8, \
                    correct_bias=False, beta1=0.9, beta2=0.999, iter=999):
        if optimization != 'adam':
            self.params['W'] -= learning_rate * (self.grads['dW'] + l2_penalty * self.params['W'])
            self.params['b'] -= learning_rate * (self.grads['db'] + l2_penalty * self.params['b'])
        else:
            if correct_bias:
                W_first_moment = self.momentum_cache['dW'] / (1 - beta1 ** iter)
                b_first_moment = self.momentum_cache['db'] / (1 - beta1 ** iter)
                W_second_moment = self.rmsprop_cache['dW'] / (1 - beta2 ** iter)
                b_second_moment = self.rmsprop_cache['db'] / (1 - beta2 ** iter)
            else:
                W_first_moment = self.momentum_cache['dW']
                b_first_moment = self.momentum_cache['db']
                W_second_moment = self.rmsprop_cache['dW']
                b_second_moment = self.rmsprop_cache['db']

            W_learning_rate = learning_rate / (np.sqrt(W_second_moment) + epsilon)
            b_learning_rate = learning_rate / (np.sqrt(b_second_moment) + epsilon)

            self.params['W'] -= W_learning_rate * (W_first_moment + l2_penalty * self.params['W'])
            self.params['b'] -= b_learning_rate * (b_first_moment + l2_penalty * self.params['b'])
