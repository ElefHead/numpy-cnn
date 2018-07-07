import numpy as np
from utilities.settings import get_layer_num, inc_layer_num


class Pooling:
    def __init__(self, kernel_shape=(3, 3), stride=1, mode="max", name=None):
        '''

        :param kernel_shape:
        :param stride:
        :param mode:
        '''
        self.params = {
            'kernel_shape': kernel_shape,
            'stride': stride,
            'mode': mode
        }
        self.type = 'pooling'
        self.cache = {}
        self.has_units = False
        self.name = name

    def has_weights(self):
        return self.has_units

    def forward_propagate(self, X, save_cache=False):
        '''

        :param X:
        :param save_cache:
        :return:
        '''

        (num_data_points, prev_height, prev_width, prev_channels) = X.shape
        filter_shape_h, filter_shape_w = self.params['kernel_shape']

        n_H = int(1 + (prev_height - filter_shape_h) / self.params['stride'])
        n_W = int(1 + (prev_width - filter_shape_w) / self.params['stride'])
        n_C = prev_channels

        A = np.zeros((num_data_points, n_H, n_W, n_C))

        for i in range(num_data_points):
            for h in range(n_H):
                for w in range(n_W):

                    vert_start = h * self.params['stride']
                    vert_end = vert_start + filter_shape_h
                    horiz_start = w * self.params['stride']
                    horiz_end = horiz_start + filter_shape_w

                    for c in range(n_C):

                        if self.params['mode'] == 'average':
                            A[i, h, w, c] = np.mean(X[i, vert_start: vert_end, horiz_start: horiz_end, c])
                        else:
                            A[i, h, w, c] = np.max(X[i, vert_start: vert_end, horiz_start: horiz_end, c])
        if save_cache:
            self.cache['A'] = X

        return A

    def distribute_value(self, dz, shape):
        (n_H, n_W) = shape
        average = 1 / (n_H * n_W)
        return np.ones(shape) * dz * average

    def create_mask(self, x):
        return x == np.max(x)

    def back_propagate(self, dA):
        A = self.cache['A']
        filter_shape_h, filter_shape_w = self.params['kernel_shape']

        (num_data_points, prev_height, prev_width, prev_channels) = A.shape
        m, n_H, n_W, n_C = dA.shape

        dA_prev = np.zeros(shape=(num_data_points, prev_height, prev_width, prev_channels))

        for i in range(num_data_points):
            a = A[i]

            for h in range(n_H):
                for w in range(n_W):

                    vert_start = h * self.params['stride']
                    vert_end = vert_start + filter_shape_h
                    horiz_start = w * self.params['stride']
                    horiz_end = horiz_start + filter_shape_w

                    for c in range(n_C):

                        if self.params['mode'] == 'average':
                            da = dA[i, h, w, c]
                            dA_prev[i, vert_start: vert_end, horiz_start: horiz_end, c] += \
                                self.distribute_value(da, self.params['kernel_shape'])

                        else:
                            a_slice = a[vert_start: vert_end, horiz_start: horiz_end, c]
                            mask = self.create_mask(a_slice)
                            dA_prev[i, vert_start: vert_end, horiz_start: horiz_end, c] += \
                                dA[i, h, w, c] * mask

        return dA_prev
