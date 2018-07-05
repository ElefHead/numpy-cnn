import numpy as np
from utils import pad_inputs
from activation import ACTIVATE as activation, D_ACTIVATE as d_activation


class Model:
    def __init__(self, filters, padding='valid', activation='relu'):
        self.params = {
            'filters': filters,
            'padding': padding,
            'activation': activation
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

    def forward_propagate(self):
        pass

    def back_propagate(self):
        pass

    def apply_grads(self):
        pass
