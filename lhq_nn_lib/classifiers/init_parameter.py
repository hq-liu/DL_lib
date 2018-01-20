import numpy as np


def constant_init(constant, shape, dtype):
    """
    constant initializer
    :param constant: number of the initializer
    :param shape: shape
    :param dtype: dtype
    :return:
    """
    return np.full(shape=shape, fill_value=constant, dtype=dtype)


def normal_init(mean, std, shape, dtype):
    """
    normal distribution initializer
    :param mean:
    :param std:
    :param shape:
    :param dtype:
    :return:
    """
    return np.random.normal(mean, std, size=shape).astype(dtype=dtype)
