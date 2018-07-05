import numpy as np
from utils import pad_inputs
from activation import ACTIVATE as activate, D_ACTIVATE as d_activate
from initializers import glorot_uniform


class Convolution:
    def __init__(self, filters, kernel_shape=(3,3), padding='valid', stride=1, activation='relu'):
        self.params = {
            'filters': filters,
            'padding': padding,
            'activation': activation,
            'kernel_shape': kernel_shape,
            'stride': stride
        }
        self.cache = {}
        self.rmsprop = {}
        self.momentum = {}
        self.grads = {}

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
        (num_data_points, prev_height, prev_width, prev_channels) = X.shape
        filter_shape_h, filter_shape_w = self.params['kernel_shape']

        if 'W' not in self.params:
            shape = (filter_shape_h, filter_shape_w, prev_channels, self.params['filters'])
            self.params['W'] = glorot_uniform(shape=shape)

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

        Z = np.zeros(shape=(num_data_points, n_H, n_W, self.params['filters']))

        X_pad = pad_inputs(X, (pad_h, pad_w))

        for i in range(num_data_points):
            x = X_pad[i]
            for h in range(n_H):
                for w in range(n_W):
                    for c in range(self.params['filters']):

                        vert_start = self.params['stride'] * h
                        vert_end = vert_start + filter_shape_h
                        horiz_start = self.params['stride'] * w
                        horiz_end = horiz_start + filter_shape_w

                        x_slice = x[vert_start: vert_end, horiz_start: horiz_end, :]

                        Z[i, h, w, c] = self.conv_single_step(x_slice, self.params['W'][:,:,:,c],
                                                              self.params['b'][:, :, :, c])

        if save_cache:
            self.cache['A'] = X
            self.cache['Z'] = Z

        return activate[self.params['activation']](Z)

    def back_propagate(self):
        pass

    def apply_grads(self):
        pass
