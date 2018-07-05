import numpy as np


def he_normal(fan_in, fan_out):
    '''
    A function for smart normal distribution based initialization of parameters
    [He et al. https://arxiv.org/abs/1502.01852]
    :param fan_in: The number of units in previous layer.
    :param fan_out: The number of units in current layer.
    :return:[numpy array, numpy array]: A randomly initialized array of shape [fan_out, fan_in]
    '''
    return np.random.normal(0, 1, size=(fan_out, fan_in)) * np.sqrt(2 / fan_out), \
           np.random.uniform(-0.2, 0.2, size=(fan_out, 1))


def he_uniform(fan_in, fan_out):
    '''
    A function for smart uniform distribution based initialization of parameters
    [He et al. https://arxiv.org/abs/1502.01852]
    :param fan_in: The number of units in previous layer.
    :param fan_out: The number of units in current layer.
    :return:[numpy array, numpy array]: A randomly initialized array of shape [fan_out, fan_in] and
            the bias of shape [fan_out, 1]
    '''
    return np.random.uniform(-1, 1, size=(fan_out, fan_in)) * np.sqrt(2 / fan_out), \
           np.random.uniform(-0.2, 0.2, size=(fan_out, fan_in))


def glorot_normal(fan_in, fan_out):
    '''
    A function for smart uniform distribution based initialization of parameters
    [Glorot et al. http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf]
    :param fan_in: The number of units in previous layer.
    :param fan_out: The number of units in current layer.
    :return:[numpy array, numpy array]: A randomly initialized array of shape [fan_out, fan_in] and
            the bias of shape [fan_out, 1]
    '''
    return np.random.normal(0, np.sqrt(2/(fan_in + fan_out + 1)), size=(fan_out, fan_in)), \
           np.random.uniform(-0.2, 0.2, size=(fan_out, 1))


def glorot_uniform(fan_in, fan_out):
    '''
    A function for smart uniform distribution based initialization of parameters
    [Glorot et al. http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf]
    :param fan_in: The number of units in previous layer.
    :param fan_out: The number of units in current layer.
    :return:[numpy array, numpy array]: A randomly initialized array of shape [fan_out, fan_in] and
            the bias of shape [fan_out, 1]
    '''
    a = np.sqrt(6/(fan_in + fan_out + 1))
    return np.random.uniform(-a, a, size=(fan_out, fan_in)), np.random.uniform(-0.2, 0.2, size=(fan_out, 1))
