import numpy as np
from layers.gradient_check import *
from layers.activation_layer import *


def affine_forward(X, W, b):
    """
    Affine forward function, y = WX+b
    :param X: inputs, shape=(N, D)
    :param W: weights, shape=(D, M)
    :param b: biases, shape=(1, M)
    :return: output(N, M) and cache=(x, w, b)
    """
    X_ = X.reshape(X.shape[0], -1)
    y = np.dot(X_, W) + b
    cache = (X, W, b)
    return y, cache


def affine_backward(dout, cache):
    """
    Affine backward function, compute the grads
    :param dout: grad of the output (N, M)
    :param cache: cache stored before (X, W, b)
    :return: dW(D, M), db(1, M), dX(N, D)
    """
    (X, W, b) = cache
    X_ = X.reshape(X.shape[0], -1)
    dX = np.dot(dout, W.T)
    dW = np.dot(X_.T, dout)
    db = np.sum(dout, axis=0, keepdims=True)

    dX = dX.reshape(X.shape)
    return dX, dW, db


if __name__ == '__main__':
    np.random.seed(231)
    x = np.random.randn(10, 2, 3)
    w = np.random.randn(6, 5)
    b = np.random.randn(5)
    dout = np.random.randn(10, 5)

    dx_num = eval_numerical_gradient_array(lambda x: affine_forward(x, w, b)[0], x, dout)
    dw_num = eval_numerical_gradient_array(lambda w: affine_forward(x, w, b)[0], w, dout)
    db_num = eval_numerical_gradient_array(lambda b: affine_forward(x, w, b)[0], b, dout)

    _, cache = affine_forward(x, w, b)
    dx, dw, db = affine_backward(dout, cache)
    print(db_num.shape)
    # The error should be around 1e-10
    print('Testing affine_backward function:')
    print('dx error: ', rel_error(dx_num, dx))
    print('dw error: ', rel_error(dw_num, dw))
    print('db error: ', rel_error(db_num, db))
