import numpy as np


class Pooling:
    def __init__(self, kernel_shape=(3, 3), stride=1, mode="max"):
        self.params = {
            'kernel_shape': kernel_shape,
            'stride': stride,
            'mode': max
        }
        self.cache = {}

    def forward_propagate(self, X, save_cache=False):
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
            self.cache['A'] = A

        return A

    def back_propagate(self):
        pass