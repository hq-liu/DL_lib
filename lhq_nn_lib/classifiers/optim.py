import numpy as np


def sgd(w, dw, learning_rate=1e-3, config=None):
    """
    stochastic gradient descent
    :param w: weights
    :param dw: grads of weights
    :param learning_rate: learning rate
    :return: new weights
    """
    w -= dw * learning_rate
    return w, config


def sgd_with_momentum(w, dw, learning_rate=1e-3, config=None):
    """
    stochastic gradient descent with momentum
    :param w: parameters
    :param dw: derivative of the parameters
    :param learning_rate: learning rate
    :param config: hyper parameter
    :return: new parameters
    """
    if config is None:
        config = {}
    config.setdefault('beta', 0.9)
    v = config.get('velocity', np.zeros_like(w))

    v = config['beta'] * v - (1 - config['beta']) * dw
    new_w = w - learning_rate * v
    return new_w, config


def RMSprop(w, dw, learning_rate=1e-3, config=None):
    """
    RMSprop optimizer
    :param w: parameters
    :param dw: derivative of the parameters
    :param learning_rate: lr
    :param config: hyper parameters and cache
    :return: new weights and new config
    """
    if config is None:
        config = {}
    config.setdefault('beta', 0.99)
    config.setdefault('epsilon', 1e-8)
    config.setdefault('S', np.zeros_like(w))

    config['S'] = config['beta'] * config['S'] + (1 - config['beta']) * (dw ** 2)
    new_w = w - learning_rate * dw / (np.sqrt(config['S']) + config['epsilon'])
    return new_w, config


def Adam(w, dw, learning_rate=1e-3, config=None):
    """
    Adam optimizer
    :param w: parameter
    :param dw: derivative of the parameters
    :param learning_rate: lr
    :param config: hyper parameters and cache
    :return: new weights and new config
    """
    if config is None:
        config = {}
    config.setdefault('beta1', 0.9)
    config.setdefault('beta2', 0.999)
    config.setdefault('epsilon', 1e-8)
    config.setdefault('t', 1)

    config.setdefault('V', np.zeros_like(w))
    config.setdefault('S', np.zeros_like(w))

    beta1, beta2, t = config['beta1'], config['beta2'], config['t']
    config['V'] = beta1 * config['V'] + (1 - beta1) * dw
    config['S'] = beta2 * config['S'] + (1 - beta2) * (dw ** 2)
    V = config['V'] / (1 - beta1 ** t)
    S = config['S'] / (1 - beta2 ** t)
    new_w = w - learning_rate * V / (np.sqrt(S) + config['epsilon'])
    config['t'] += 1
    return new_w, config


