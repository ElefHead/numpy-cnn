import matplotlib.pyplot as plt
import numpy as np


labels_to_name_map = {
    0: 'airplane',
    1: 'automobile',
    2: 'bird',
    3: 'cat',
    4: 'deer',
    5: 'dog',
    6: 'frog',
    7: 'horse',
    8: 'ship',
    9: 'truck'
}


def get_name(label):
    return labels_to_name_map[int(np.argmax(label))]


def pad_inputs(X, pad):
    '''
    Function to apply zero padding to the image
    :param X:[numpy array]: Dataset of shape (m, height, width, depth)
    :param pad:[int]: number of columns to pad
    :return:[numpy array]: padded dataset
    '''
    return np.pad(X, ((0, 0), (pad[0], pad[0]), (pad[1], pad[1]), (0, 0)), 'constant')


def show_image(image, title=None, cmap=None):
    '''
    Function to display one image
    :param image: numpy float array: of shape (32, 32, 3)
    :return: Void
    '''
    if cmap is not None:
        plt.imshow(image, cmap=cmap)
    else:
        plt.imshow(image)
    if title is not None:
        plt.title(title)
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
    N = data.shape[1] if len(data.shape) == 2 else data.shape[0]
    num_batches = N//batch_size
    if len(data.shape) == 2:
        data = data.T
    if shuffle:
        shuffled_indices = np.random.permutation(N)
        data = data[shuffled_indices]
        labels = labels[:, shuffled_indices] if labels is not None else None
    if num_batches == 0:
        if labels is not None:
            yield (data.T, labels) if len(data.shape) == 2 else (data, labels)
        else:
            yield data.T if len(data.shape) == 2 else data
    for batch_num in range(num_batches):
        if labels is not None:
            yield (data[batch_num*batch_size:(batch_num+1)*batch_size].T,
                  labels[:, batch_num*batch_size:(batch_num+1)*batch_size]) if len(data.shape) == 2 \
                      else (data[batch_num*batch_size:(batch_num+1)*batch_size],
                  labels[:, batch_num*batch_size:(batch_num+1)*batch_size])
        else:
            yield data[batch_num*batch_size:(batch_num+1)*batch_size].T if len(data.shape) == 2 else \
                data[batch_num*batch_size:(batch_num+1)*batch_size]
    if N%batch_size != 0 and num_batches != 0:
        if labels is not None:
            yield (data[num_batches*batch_size:].T, labels[:, num_batches*batch_size:]) if len(data.shape) == 2 else \
                (data[num_batches*batch_size:], labels[:, num_batches*batch_size:])
        else:
            yield data[num_batches*batch_size:].T if len(data.shape)==2 else data[num_batches*batch_size:]


def evaluate(labels, predictions):
    '''
    A function to compute the accuracy of the predictions on a scale of 0-1.
    :param labels:[numpy array]: Training labels (or testing/validation if available)
    :param predictions:[numpy array]: Predicted labels
    :return:[float]: a number between [0, 1] denoting the accuracy of the prediction
    '''
    return np.mean(np.argmax(labels, axis=0) == np.argmax(predictions, axis=0))
