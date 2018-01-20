import numpy as np
from layers.dense_layer import *
from layers.activation_layer import *
from layers.loss import *
from classifiers.init_parameter import *
from layers.layer_utils import *


class Two_layers_dense_net():
    def __init__(self, input_dim, hidden_dim, num_classes, reg):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.reg = reg
        self.params = {}
        self.cache = {}
        self.grad = {}

        self.params['W1'] = normal_init(0, 0.01,
                                        shape=[input_dim, hidden_dim],
                                        dtype=np.float32)
        self.params['b1'] = constant_init(0.1, shape=[1, hidden_dim],
                                          dtype=np.float32)
        self.params['W2'] = normal_init(0, 0.01,
                                        shape=[hidden_dim, num_classes],
                                        dtype=np.float32)
        self.params['b2'] = constant_init(0.1, shape=[1, num_classes],
                                          dtype=np.float32)

    def forward(self, inputs):
        hidden_state, self.cache['layer_1'] = affine_relu_forward(inputs,
                                                      self.params['W1'], self.params['b1'])
        output, self.cache['layer_2'] = affine_forward(hidden_state,
                                                  self.params['W2'], self.params['b2'])
        return output

    def backward(self, dout):
        dh, self.grad['W2'], self.grad['b2'] = affine_backward(dout, self.cache['layer_2'])
        dx, self.grad['W1'], self.grad['b1'] = affine_relu_backward(dh, self.cache['layer_1'])
        self.grad['W1'] += self.reg * self.params['W1']
        self.grad['W2'] += self.reg * self.params['W2']
        return dx, self.grad

    def compute_loss(self, outputs, labels):
        W1, W2 = self.params['W1'], self.params['W2']

        loss, d_loss = softmax_loss(outputs, labels)
        total_loss = loss + 0.5 * self.reg * (np.sum(W1 ** 2)+np.sum(W2 ** 2))
        return total_loss, d_loss


class Multi_layer_fc():
    def __init__(self, input_dim, layer_num,
                 hidden_dim, num_class=10, use_batchnorm=False,
                 dropout=0, dtypt=np.float32, seed=None):
        """
        multi_layer fully connected network
        :param input_dim: input dim
        :param layer_num: number of the hidden layers
        :param hidden_dim: hidden layer's dim
        :param num_class: number of classes
        :param use_batchnorm: use batchnorm or not
        :param dropout: use dropout or not
        :param dtypt:
        :param seed:
        """
        self.input_dim = input_dim
        self.layer_num = layer_num
        self.hidden_dim = hidden_dim
        self.num_classes = num_class
        self.use_batchnorm = use_batchnorm
        self.dropout = dropout
        self.dtype = dtypt
        self.seed = seed

        self.params = {}
        self.cache = {}
        self.grads = {}

        self.params['W1'] = normal_init(0, 0.01,
                                        shape=[input_dim, hidden_dim],
                                        dtype=self.dtype)
        self.params['b1'] = constant_init(0.1, shape=[1, hidden_dim],
                                          dtype=self.dtype)
        for i in range(2, layer_num):
            self.params['W'+str(i)] = normal_init(0, 0.01,
                                                  shape=[hidden_dim, hidden_dim],
                                                  dtype=self.dtype)
            self.params['b'+str(i)] = constant_init(0.1, shape=[1, hidden_dim],
                                                    dtype=self.dtype)
            if self.use_batchnorm:
                self.params['gamma'+str(i)] = np.ones(shape=[1, hidden_dim], dtype=self.dtype)
                self.params['beta'+str(i)] = np.zeros_like(self.params['gamma'+str(i)])
        self.params['W'+str(layer_num)] = normal_init(0, 0.01,
                                                  shape=[hidden_dim, num_class],
                                                  dtype=self.dtype)
        self.params['b'+str(layer_num)] = constant_init(0.1,
                                                        shape=[1, num_class],
                                                        dtype=self.dtype)
        if self.use_batchnorm:
            self.params['gamma' + str(layer_num)] = np.ones(shape=[1, hidden_dim], dtype=self.dtype)
            self.params['beta' + str(layer_num)] = np.zeros_like(self.params['gamma' + str(layer_num)])

    def forward(self, inputs):
        pass

