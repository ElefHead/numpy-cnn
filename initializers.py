import numpy as np


def he_initialize(fan_in, fan_out):
    '''
    A function from smart initialization of parameters [He et al. https://arxiv.org/abs/1502.01852]
    :param fan_in: The number of units in previous layer.
    :param fan_out: The number of units in current layer.
    :return:[numpy array]: A randomly initialized array of shape [fan_out, fan_in]
    '''
    return np.random.normal(0, 1, size=(fan_out, fan_in)) * np.sqrt(2 / fan_out), \
           np.random.uniform(-0.2, 0.2, size=(fan_out, 1))


