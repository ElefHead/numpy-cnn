import matplotlib.pyplot as plt
import numpy as np


def show_image(image, cmap=None):
    '''
    Function to display one image
    :param image: numpy float array: of shape (32, 32, 3)
    :return: Void
    '''
    if cmap is not None:
        plt.imshow(image, cmap=cmap)
    else:
        plt.imshow(image)
    plt.show()


def plot_graph(Y, X=None, title=None, xlabel=None, ylabel=None):
    '''
    A function to plot a line graph.
    :param Y: Values for Y axis
    :param X: Values for X axis(optional)
    :param title:[string default=None]: Graph title.
    :param xlabel:[string default=None]: X axis label.
    :param ylabel:[string default=None]: Y axis label.
    :return: Void
    '''
    if X is None:
        plt.plot(Y)
    else:
        plt.plot(X, Y)
    if title is not None:
        plt.title(title)
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    plt.show()


def to_categorical(labels, num_classes, axis=0):
    '''
    Function to one-hot-encode the labels
    :param labels:[list or vector]: list of ints: list of numbers (ranging 0-9 for CIFAR-10)
    :param num_classes:[int]: the total number of unique classes or categories.
    :param axis:[int Default=0]: decides row matrix or column matrix. if 0 then column matrix, else row
    :return: numpy array of ints: one-hot-encoded labels
    '''
    ohe_labels = np.zeros((len(labels), num_classes)) if axis != 0 else np.zeros((num_classes, len(labels)))
    for _ in range(len(labels)):
        if axis == 0:
            ohe_labels[labels[_], _] = 1
        else:
            ohe_labels[_, labels[_]] = 1
    return ohe_labels


def get_batches(data, labels=None, batch_size=256, shuffle=True):
    '''
    Function to get data in batches.
    :param data:[numpy array]: training or test data. Assumes shape=[M, N] where M is the features and N is samples.
    :param labels:[numpy array, Default = None (for without labels)]: actual labels corresponding to the data.
    Assumes shape=[M, N] where M is number of classes/results per sample and N is number of samples.
    :param batch_size:[int, Default = 256]: required size of batch. If data can't be exactly divided by batch_size,
    remaining samples will be in a new batch
    :param shuffle:[boolean, Default = True]: if true, function will shuffle the data
    :return:[numpy array, numpy array]: batch data and corresponding labels
    '''
    M, N = data.shape
    num_batches = N//batch_size
    if shuffle:
        shuffled_indices = np.random.permutation(N)
        data = data[:, shuffled_indices]
        labels = labels[:, shuffled_indices] if labels is not None else None
    if num_batches == 0:
        if labels is not None:
            yield data, labels
        else:
            yield data
    for batch_num in range(num_batches):
        if labels is not None:
            yield data[:, batch_num*batch_size:(batch_num+1)*batch_size], \
                  labels[:, batch_num*batch_size:(batch_num+1)*batch_size]
        else:
            yield data[:, batch_num*batch_size:(batch_num+1)*batch_size]
    if N%batch_size != 0 and num_batches != 0:
        if labels is not None:
            yield data[:, num_batches*batch_size:], labels[:, num_batches*batch_size:]
        else:
            yield data[:, num_batches*batch_size:]


def evaluate(labels, predictions):
    '''
    A function to compute the accuracy of the predictions on a scale of 0-1.
    :param labels:[numpy array]: Training labels (or testing/validation if available)
    :param predictions:[numpy array]: Predicted labels
    :return:[float]: a number between [0, 1] denoting the accuracy of the prediction
    '''
    return np.mean(np.argmax(labels, axis=0) == np.argmax(predictions, axis=0))


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


def cross_entropy_loss(labels, predictions, epsilon=1e-8):
    '''
    The function to compute the categorical cross entropy loss, given training labels and prediction
    :param labels:[numpy array]: Training labels
    :param predictions:[numpy array]: Predicted labels
    :param epsilon:[float default=1e-8]: A small value for applying clipping for stability
    :return:[float]: The computed value of loss.
    '''
    predictions /= np.sum(predictions, axis=0, keepdims=True)
    predictions = np.clip(predictions, epsilon, 1. - epsilon)
    return -np.sum(labels * np.log(predictions))


def d_cross_entropy_loss(labels, predictions):
    '''
    The function to compute the derivative values of categorical cross entropy values, given labels and prediction
    :param labels:[numpy array]: Training labels
    :param predictions:[numpy array]: Predicted labels
    :return:[numpy array]: The computed derivatives of categorical cross entropy function.
    '''
    return labels - predictions
