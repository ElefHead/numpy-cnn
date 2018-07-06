import pickle
import numpy as np
from os import path
from utilities.utils import to_categorical


TOTAL_BATCHES = 5
NUM_DIMENSIONS = 3072
NUM_CLASSES = 10
SAMPLES_PER_BATCH = 10000
MAX_TRAINING_SAMPLES = 50000
MAX_TESTING_SAMPLES = 10000
FILE_NAME = {
    'training': 'data_batch_',
    'testing': 'test_batch'
}


def unpickle(file, num_samples=10000):
    '''
    Function to read the data from the binary files
    Description of data taken from CIFAR-10 website
    :param file: the path to the datafile
    :param num_samples: (remaining) samples required from a particular set (not same as num_samples in get_data)
    :return: data and one-hot-encoded labels
    '''
    with open(file, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
    return data[b'data'][:num_samples, :], to_categorical(data[b'labels'][:num_samples], NUM_CLASSES)


def get_data(data_path="data", num_samples=50000, dataset="training"):
    '''
    Function that reads and returns the required training or testing data
    :param data_path: string: the relative folder path to where the data lies (default: ./data)
    :param num_samples: int: number of samples required (MAX 50000)
    :param dataset: string: training or testing, default is training
    :return: two numpy arrays 1 containing data and other containing corresponding labels.
             data shape = [num_samples, 32, 32, 3] and labels shape = [num_samples, 10] for cifar-10 data
             consistency checked with keras dataset cifar10
    '''
    if dataset == "testing" and num_samples > MAX_TESTING_SAMPLES:
        num_samples = MAX_TESTING_SAMPLES
    if dataset == "training" and num_samples>MAX_TRAINING_SAMPLES:
        num_samples = MAX_TRAINING_SAMPLES
    data = np.zeros(shape=(num_samples, NUM_DIMENSIONS))
    labels = np.zeros(shape=(NUM_CLASSES, num_samples))
    num_batches = num_samples//SAMPLES_PER_BATCH + 1
    if num_batches > TOTAL_BATCHES:
        num_batches = TOTAL_BATCHES
    remaining = num_samples - 0
    for _ in range(num_batches):
        file_name = FILE_NAME[dataset]+str(_+1) if dataset=="training" else FILE_NAME[dataset]
        file = path.join('.', data_path, file_name)
        if remaining > SAMPLES_PER_BATCH:
            ret_val = unpickle(file, SAMPLES_PER_BATCH)
            data[_*SAMPLES_PER_BATCH: SAMPLES_PER_BATCH*(_+1)] = ret_val[0]
            labels[:, _*SAMPLES_PER_BATCH: SAMPLES_PER_BATCH*(_+1)] = ret_val[1]
        else:
            ret_val = unpickle(file, remaining)
            data[_*SAMPLES_PER_BATCH:] = ret_val[0]
            labels[:, _*SAMPLES_PER_BATCH:] = ret_val[1]
        remaining = remaining - SAMPLES_PER_BATCH
    return data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).astype(np.float32), labels.T
