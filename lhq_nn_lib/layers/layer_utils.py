import numpy as np
from layers.activation_layer import *
from layers.dense_layer import *
from layers.gradient_check import *


def affine_relu_forward(X, W, b):
    y, affine_cache = affine_forward(X, W, b)
    outputs, relu_cache = relu_forward(y)
    cache = (affine_cache, relu_cache)
    return outputs, cache


def affine_relu_backward(dout, cache):
    affine_cache, relu_cache = cache
    dy = relu_backward(dout, relu_cache)
    dx, dw, db = affine_backward(dy, affine_cache)
    return dx, dw, db


if __name__ == '__main__':
    np.random.seed(231)
    x = np.random.randn(2, 3, 4)
    w = np.random.randn(12, 10)
    b = np.random.randn(10)
    dout = np.random.randn(2, 10)

    out, cache = affine_relu_forward(x, w, b)
    dx, dw, db = affine_relu_backward(dout, cache)

    dx_num = eval_numerical_gradient_array(lambda x: affine_relu_forward(x, w, b)[0], x, dout)
    dw_num = eval_numerical_gradient_array(lambda w: affine_relu_forward(x, w, b)[0], w, dout)
    db_num = eval_numerical_gradient_array(lambda b: affine_relu_forward(x, w, b)[0], b, dout)

    print('Testing affine_relu_forward:')
    print('dx error: ', rel_error(dx_num, dx))
    print('dw error: ', rel_error(dw_num, dw))
    print('db error: ', rel_error(db_num, db))
