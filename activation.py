import numpy as np


def softmax(Z):
    '''
    A function to compute the softmax activation
    :param Z:[numpy array]: Array of floats
    :return:[numpy array]: Array of floats, after application of softmax function to Z
    '''
    Z_ = Z - Z.max()
    e = np.exp(Z_)
    return e / np.sum(e, axis=0, keepdims=True)


def d_softmax(Z):
    '''
    A function to compute the derivative values of softmax activation
    :param Z:[numpy array]: Array of floats
    :return:[numpy array]: Array of floats, values corresponding to the derivative of softmax activation on Z
    '''
    return Z * (1 - Z)


def relu(Z):
    '''
    A function to compute the relu(rectified linear unit) activation values.
    :param Z:[numpy array]: Array of floats, the score values.
    :return:[numpy array]: relu activated values.
    '''
    return np.where(Z >= 0, Z, 0)


def d_relu(Z):
    '''
    A function to compute the derivative of elu(exponential linear unit) activation values.
    :param Z:[numpy array]: Array of floats, the score values.
    :return:[numpy array]: the required derivative values
    '''
    return np.where(Z >= 0, 1, 0)


def elu(Z, alpha=1.2):
    '''
    A function to compute the elu(exponential linear unit) activation values.
    :param Z:[numpy array]: Array of floats, the score values.
    :param alpha:[float default=1.2]: the value for elu alpha
    :return:[numpy array]: elu activated values
    '''
    return np.where(Z >= 0, Z, alpha * (np.exp(Z) - 1))


def d_elu(Z, alpha=1.2):
    '''
    A function to compute the derivative of elu(exponential linear unit) activation values.
    :param Z:[numpy array]: Array of floats, the score values.
    :param alpha:[float default=1.2]: the value for elu alpha
    :return:[numpy array]: the required derivative values
    '''
    return np.where(Z >= 0, 1, elu(Z, alpha) + alpha)


def selu(Z, alpha=1.6733, selu_lambda=1.0507):
    '''
    A function to compute the scaled exponential linear unit
    activation value. [Klambauer et al. https://arxiv.org/abs/1706.02515]
    :param Z:[numpy array]: Array of floats, the score values.
    :param alpha:[float default=1.6733]: the value for selu alpha
    :param selu_lambda:[float default=1.0507]: the value for selu lambda
    :return:[numpy array] selu activated values
    '''
    return selu_lambda*np.where(Z >= 0, Z, alpha*(np.exp(Z) - 1))


def d_selu(Z, alpha=1.6733, selu_lambda=1.0507):
    '''
    A function to compute the derivative of selu
    :param Z:[numpy array]: Array of floats, the score values.
    :param alpha:[float default=1.6733]: the value for selu alpha
    :param selu_lambda:[float default=1.0507]: the value for selu lambda
    :return:[numpy array]: required derivative values
    '''
    return selu_lambda*np.where(Z >= 0, 1, alpha*np.exp(Z))


ACTIVATE = {
    'elu': elu,
    'selu': selu,
    'softmax': softmax,
    'relu': relu
}

D_ACTIVATE = {
    'elu': d_elu,
    'selu': d_selu,
    'softmax': d_softmax,
    'relu': d_relu
}