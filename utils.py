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


def to_categorical(labels, num_classes):
    '''
    Function to one-hot-encode the labels
    :param labels: list of ints: list of numbers (ranging 0-9 for CIFAR-10)
    :return: numpy array of ints: one-hot-encoded labels
    '''
    ohe_labels = np.zeros((len(labels), num_classes))
    for _ in range(len(labels)):
        ohe_labels[_, labels[_]] = 1
    return ohe_labels