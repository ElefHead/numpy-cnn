import numpy as np


def get_fans(shape):
    '''

    :param shape:
    :return:
    '''
    fan_in = shape[0] if len(shape) == 2 else np.prod(shape[1:])
    fan_out = shape[1] if len(shape) == 2 else shape[0]
    return fan_in, fan_out


def normal(shape, scale=0.05):
    '''

    :param shape:
    :param scale:
    :return:
    '''
    return np.random.normal(0, scale, size=shape)


def uniform(shape, scale=0.05):
    '''

    :param shape:
    :param scale:
    :return:
    '''
    return np.random.uniform(-scale, scale, size=shape)


def he_normal(shape):
    '''
    A function for smart normal distribution based initialization of parameters
    [He et al. https://arxiv.org/abs/1502.01852]
    :param fan_in: The number of units in previous layer.
    :param fan_out: The number of units in current layer.
    :return:[numpy array, numpy array]: A randomly initialized array of shape [fan_out, fan_in]
    '''
    fan_in, fan_out = get_fans(shape)
    scale = np.sqrt(2. / fan_in)
    shape = (fan_out, fan_in) if len(shape) == 2 else shape         # For a fully connected network
    bias_shape = (fan_out, 1) if len(shape) == 2 else (
        1, 1, 1, shape[3])   # This supports only CNNs and fully connected networks
    return normal(shape, scale), uniform(bias_shape)


def he_uniform(shape):
    '''
    A function for smart uniform distribution based initialization of parameters
    [He et al. https://arxiv.org/abs/1502.01852]
    :param fan_in: The number of units in previous layer.
    :param fan_out: The number of units in current layer.
    :return:[numpy array, numpy array]: A randomly initialized array of shape [fan_out, fan_in] and
            the bias of shape [fan_out, 1]
    '''
    fan_in, fan_out = get_fans(shape)
    scale = np.sqrt(6. / fan_in)
    shape = (fan_out, fan_in) if len(shape) == 2 else shape  # For a fully connected network
    bias_shape = (fan_out, 1) if len(shape) == 2 else (
        1, 1, 1, shape[3])  # This supports only CNNs and fully connected networks
    return uniform(shape, scale), uniform(shape=bias_shape)


def glorot_normal(shape):
    '''
    A function for smart uniform distribution based initialization of parameters
    [Glorot et al. http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf]
    :param fan_in: The number of units in previous layer.
    :param fan_out: The number of units in current layer.
    :return:[numpy array, numpy array]: A randomly initialized array of shape [fan_out, fan_in] and
            the bias of shape [fan_out, 1]
    '''
    fan_in, fan_out = get_fans(shape)
    scale = np.sqrt(2. / (fan_in + fan_out))
    shape = (fan_out, fan_in) if len(shape) == 2 else shape  # For a fully connected network
    bias_shape = (fan_out, 1) if len(shape) == 2 else (
        1, 1, 1, shape[3])  # This supports only CNNs and fully connected networks
    return normal(shape, scale), uniform(shape=bias_shape)


def glorot_uniform(shape):
    '''
    A function for smart uniform distribution based initialization of parameters
    [Glorot et al. http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf]
    :param fan_in: The number of units in previous layer.
    :param fan_out: The number of units in current layer.
    :return:[numpy array, numpy array]: A randomly initialized array of shape [fan_out, fan_in] and
            the bias of shape [fan_out, 1]
    '''
    fan_in, fan_out = get_fans(shape)
    scale = np.sqrt(6. / (fan_in + fan_out))
    shape = (fan_out, fan_in) if len(shape) == 2 else shape  # For a fully connected network
    bias_shape = (fan_out, 1) if len(shape) == 2 else (
        1, 1, 1, shape[3])  # This supports only CNNs and fully connected networks
    return uniform(shape, scale), uniform(shape=bias_shape)
