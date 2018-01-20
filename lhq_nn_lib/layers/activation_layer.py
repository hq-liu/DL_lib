import numpy as np


def sigmoid_forward(x):
    """
    Sigmoid function, sigmoid(x) = 1/(1+exp(-x))
    :param x: input
    :return: output and cache
    """
    return 1 / (1 + np.exp(-x)), x


def sigmoid_backward(dout, cache):
    """
    backward of sigmoid function, dx = x(x-1)
    :param dout: grad of outputs
    :param cache: cache stored before
    :return: dx
    """
    x = cache
    return x * (x - 1) * dout


def tanh_forward(x):
    """
    Tanh function, tanh(x) = (exp(x) - exp(-x))/ (exp(x) + exp(-x))
    :param x: input
    :return: output
    """
    y = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
    return y, y


def tanh_backward(dout, cache):
    """
    backward of tanh, dx = 1 - (tanh(x))^2
    :param dout:
    :param cache:
    :return:
    """
    tanh_x = cache
    return (1 - tanh_x ** 2) * dout


def softmax_forward(x):
    """
    Softmax function, softmax(x) = exp(x[i]) / sum(exp(x[i]))
    :param x: input
    :return: output
    """
    temp = np.sum(np.exp(x))
    return np.exp(x) / temp, x


def softmax_backward(dout, cache):
    """
    backward of softmax, if i=j dx=x[j](1-x[j]) if i!=j dx=-x[i]x[j]
    x.shape=(N, 1)
    :param dout: grad of outputs
    :return: grads of input
    """
    x = cache
    dx_1 = x * (1 - x)
    # sum_x_2 = np.sum(np.dot(x.T, x), axis=0, keepdims=True)
    # sum_x_2 -= x ** 2
    dx = dx_1 * dout

    return dx


def relu_forward(x):
    """
    ReLu function, relu(x) = max(0, x)
    :param x: input
    :return: output and cache
    """
    return np.maximum(0, x), x


def relu_backward(dout, cache):
    x = cache
    dx = dout
    dx[x <= 0] = 0
    return dx


def leaky_relu_forward(x, alpha=0.01):
    """
    Leaky ReLu function, leaky_relu(x)=max(alpha * x, x)
    :param x: input
    :param alpha: hyperparameter of leaky relu
    :return: output
    """
    return np.maximum(alpha * x, x)


if __name__ == '__main__':
    # x = np.random.randn(1, 10)
    # y, cache = relu_forward(x)
    # dy = np.random.randn(1, 10)
    # dx = relu_backward(dy, cache)
    # print(dx)
    x = np.ones((1, 2))
    y = np.array([1,2]).reshape(1,2)
    print(x[y<2])
