import numpy as np


def dropout_forward(x, keep_prob=0.5, is_training=True):
    """
    dropout forward function
    :param x: inputs
    :param keep_prob: keep probability
    :param is_training: training or not
    :return: outputs
    """
    cache = None
    if is_training:
        mask = np.random.rand(*x.shape) < (1-keep_prob)
        cache = mask
        out = mask * x
    else:
        out = x
    return out, cache


def dropout_backward(dout, cache, is_training=True):
    """
    backward of dropout
    :param dout: derivitative of the output
    :param cache: cache
    :param is_training:
    :return: grad of x
    """
    if is_training:
       mask = cache
       dx = mask * dout
    else:
       dx = dout

    return dx

