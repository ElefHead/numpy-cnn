import matplotlib.pyplot as plt
import numpy as np


def show_image(image):
    '''
    Function to display one image
    :param image: numpy float array: of shape (32, 32, 3)
    :return: None. Plots the image
    '''
    plt.imshow(image)
    plt.show()


def to_categorical(labels, num_classes, axis=0):
    '''
    Function to one-hot-encode the labels
    :param labels:[list or vector]: list of ints: list of numbers (ranging 0-9 for CIFAR-10)
    :param num_classes:[int]: the total number of unique classes or categories.
    :param axis:[int Default=0]: decides row matrix or column matrix. if 0 then column matrix, else row
    :return: numpy array of ints: one-hot-encoded labels
    '''
    ohe_labels = np.zeros((len(labels), num_classes)) if axis == 0 else np.zeros((num_classes, len(labels)))
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
