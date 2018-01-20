import pickle
import os
import numpy as np


def load_pickle(file):
    return pickle.load(file, encoding='latin1')


def load_cifar10_batch(filename):
    with open(filename, 'rb') as file:
        datadict = load_pickle(file)
        X = datadict['data']
        Y = datadict['labels']
        X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype('float')
        Y = np.array(Y)
        return X, Y

def load_cifar10(root):
    xs = []
    ys = []
    for i in range(1, 6):
        file = os.path.join(root, 'data_batch_' + str(i))
        X, Y = load_cifar10_batch(file)
        xs.append(X)
        ys.append(Y)
    X_train = np.concatenate(xs)
    Y_train = np.concatenate(ys)
    X_test, Y_test = load_cifar10_batch(os.path.join(root, 'test_batch'))
    return X_train, Y_train, X_test, Y_test


if __name__ == '__main__':
    cwd = os.getcwd()
    root = os.path.join(cwd, 'cifar-10-batches-py')
    X_train, Y_train, X_test, Y_test = load_cifar10(root)
