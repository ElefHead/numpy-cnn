import numpy as np
import pickle
from os import path, makedirs, remove

from utilities.utils import pad_inputs
from utilities.initializers import glorot_uniform
from utilities.settings import get_layer_num, inc_layer_num


class Convolution:
    def __init__(self, filters, kernel_shape=(3, 3), padding='valid', stride=1, name=None):
        self.params = {
            'filters': filters,
            'padding': padding,
            'kernel_shape': kernel_shape,
            'stride': stride
        }
        self.cache = {}
        self.rmsprop_cache = {}
        self.momentum_cache = {}
        self.grads = {}
        self.has_units = True
        self.name = name
        self.type = 'conv'

    def has_weights(self):
        return self.has_units

    def save_weights(self, dump_path):
        dump_cache = {
            'cache': self.cache,
            'grads': self.grads,
            'momentum': self.momentum_cache,
            'rmsprop': self.rmsprop_cache
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
        self.rmsprop_cache = dump_cache['rmsprop']

    def conv_single_step(self, input, W, b):
        '''
        Function to apply one filter to input slice.
        :param input:[numpy array]: slice of input data of shape (f, f, n_C_prev)
        :param W:[numpy array]: One filter of shape (f, f, n_C_prev)
        :param b:[numpy array]: Bias value for the filter. Shape (1, 1, 1)
        :return:
        '''
        return np.sum(np.multiply(input, W)) + float(b)

    def forward_propagate(self, X, save_cache=False):
        '''

        :param X:
        :param save_cache:
        :return:
        '''
        if self.name is None:
            self.name = '{}_{}'.format(self.type, get_layer_num(self.type))
            inc_layer_num(self.type)

        (num_data_points, prev_height, prev_width, prev_channels) = X.shape
        filter_shape_h, filter_shape_w = self.params['kernel_shape']

        if 'W' not in self.params:
            shape = (filter_shape_h, filter_shape_w, prev_channels, self.params['filters'])
            self.params['W'], self.params['b'] = glorot_uniform(shape=shape)

        if self.params['padding'] == 'same':
            pad_h = int(((prev_height - 1)*self.params['stride'] + filter_shape_h - prev_height) / 2)
            pad_w = int(((prev_width - 1)*self.params['stride'] + filter_shape_w - prev_width) / 2)
            n_H = prev_height
            n_W = prev_width
        else:
            pad_h = 0
            pad_w = 0
            n_H = int((prev_height - filter_shape_h) / self.params['stride']) + 1
            n_W = int((prev_width - filter_shape_w) / self.params['stride']) + 1

        self.params['pad_h'], self.params['pad_w'] = pad_h, pad_w

        Z = np.zeros(shape=(num_data_points, n_H, n_W, self.params['filters']))

        X_pad = pad_inputs(X, (pad_h, pad_w))

        for i in range(num_data_points):
            x = X_pad[i]
            for h in range(n_H):
                for w in range(n_W):
                    vert_start = self.params['stride'] * h
                    vert_end = vert_start + filter_shape_h
                    horiz_start = self.params['stride'] * w
                    horiz_end = horiz_start + filter_shape_w

                    for c in range(self.params['filters']):

                        x_slice = x[vert_start: vert_end, horiz_start: horiz_end, :]

                        Z[i, h, w, c] = self.conv_single_step(x_slice, self.params['W'][:, :, :, c],
                                                              self.params['b'][:, :, :, c])

        if save_cache:
            self.cache['A'] = X

        return Z

    def back_propagate(self, dZ):
        '''

        :param dZ:
        :return:
        '''
        A = self.cache['A']
        filter_shape_h, filter_shape_w = self.params['kernel_shape']
        pad_h, pad_w = self.params['pad_h'], self.params['pad_w']

        (num_data_points, prev_height, prev_width, prev_channels) = A.shape

        dA = np.zeros((num_data_points, prev_height, prev_width, prev_channels))
        self.grads = self.init_cache()

        A_pad = pad_inputs(A, (pad_h, pad_w))
        dA_pad = pad_inputs(dA, (pad_h, pad_w))

        for i in range(num_data_points):
            a_pad = A_pad[i]
            da_pad = dA_pad[i]

            for h in range(prev_height):
                for w in range(prev_width):

                    vert_start = self.params['stride'] * h
                    vert_end = vert_start + filter_shape_h
                    horiz_start = self.params['stride'] * w
                    horiz_end = horiz_start + filter_shape_w

                    for c in range(self.params['filters']):
                        a_slice = a_pad[vert_start: vert_end, horiz_start: horiz_end, :]

                        da_pad[vert_start:vert_end, horiz_start:horiz_end, :] += self.params['W'][:, :, :, c] * dZ[i, h, w, c]
                        self.grads['dW'][:, :, :, c] += a_slice * dZ[i, h, w, c]
                        self.grads['db'][:, :, :, c] += dZ[i, h, w, c]
            dA[i, :, :, :] = da_pad[pad_h: -pad_h, pad_w: -pad_w, :]

        return dA

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

    def apply_grads(self, learning_rate=0.001, l2_penalty=1e-4, optimization='adam', epsilon=1e-8,
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
