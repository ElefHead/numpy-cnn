import numpy as np
import pickle
from os import path, makedirs, remove

from utilities.initializers import he_normal
from utilities.settings import get_layer_num, inc_layer_num

np.random.seed(0)


class FullyConnected:
    def __init__(self, units=200, name=None):
        self.units = units
        self.params = {}
        self.cache = {}
        self.grads = {}
        self.momentum_cache = {}
        self.rmsprop_cache = {}
        self.has_units = True
        self.type = 'fc'
        self.name = name

    def has_weights(self):
        return self.has_units

    def save_weights(self, dump_path):
        dump_cache = {
            'cache': self.cache,
            'grads': self.grads,
            'momentum': self.momentum_cache,
            'rmsprop_cache': self.rmsprop_cache
        }
        save_path = path.join(dump_path, self.name+'.pickle')
        makedirs(path.dirname(save_path), exist_ok=True)
        remove(save_path)
        with open(save_path, 'wb') as d:
            pickle.dump(dump_cache, d)

    def load_weights(self, dump_path):
        if self.name is None:
            self.name = '{}_{}'.format(self.type, get_layer_num(self.type))
            inc_layer_num(self.type)
        read_path = path.join(dump_path, self.name+'.pickle')
        with open(read_path, 'rb') as r:
            dump_cache = pickle.load(r)
        self.cache = dump_cache['cache']
        self.grads = dump_cache['grads']
        self.momentum_cache = dump_cache['momentum']
        self.rmsprop_cache = dump_cache['rmsprop_cache']

    def forward_propagate(self, X, save_cache=False):
        if self.name is None:
            self.name = '{}_{}'.format(self.type, get_layer_num(self.type))
            inc_layer_num(self.type)

        if 'W' not in self.params:
            self.params['W'], self.params['b'] = he_normal((X.shape[0], self.units))
        Z = np.dot(self.params['W'], X) + self.params['b']
        if save_cache:
            self.cache['A'] = X
        return Z

    def back_propagate(self, dZ):
        batch_size = dZ.shape[1]
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
