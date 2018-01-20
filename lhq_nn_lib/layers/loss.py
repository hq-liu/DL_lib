import numpy as np
from layers.activation_layer import *
from layers.gradient_check import *


def mean_square_error_loss(y_hat, y):
    """
    MSE loss, loss=mean(y_hat-y)^2
    :param y_hat: output of the network
    :param y: input labels
    :return: MSE loss
    """
    loss = np.mean((y_hat - y) ** 2)
    num_output = y.shape[1]
    d_loss = 2 * (y_hat - y) / num_output
    return loss, d_loss


def cross_entropy_loss(y_hat, y):
    """
    Cross entropy loss, loss = -sum(yi * log(y_hat))
    :param y_hat: output of the network
    :param y: input labels (one_hot)
    :return: cross entropy loss
    """
    loss = -np.sum(y * np.log(y_hat), axis=1)
    # loss = np.mean(loss, axis=0)
    d_loss = -y / y_hat
    return loss, d_loss


def softmax_loss(x, y):
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    N = x.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx


if __name__ == '__main__':
    np.random.seed(231)
    num_classes, num_inputs = 10, 50
    x = 0.001 * np.random.randn(num_inputs, num_classes)
    y = np.random.randint(num_classes, size=num_inputs)

    dx_num = eval_numerical_gradient(lambda x: softmax_loss(x, y)[0], x, verbose=False)
    loss, dx = softmax_loss(x, y)

    # Test softmax_loss function. Loss should be 2.3 and dx error should be 1e-8
    print('\nTesting softmax_loss:')
    print('loss: ', loss)
    print('dx error: ', rel_error(dx_num, dx))

