import numpy as np
from datasets.data_utils import *
from classifiers.fc_net import *
from classifiers.optim import *
from classifiers.solver import *


if __name__ == '__main__':
    project_dir = '/home/lhq/PycharmProjects/lhq_nn_lib/'
    root = os.path.join(project_dir, 'datasets/cifar-10-batches-py')
    logs = os.path.join(project_dir, 'logs/two_layer_fc')
    check_point = logs + '/model'
    X_train, Y_train, X_test, Y_test = load_cifar10(root)
    batch_size = 64
    lr = 1e-3
    X_train, Y_train, X_test, Y_test = X_train.reshape(50000, 3072), \
                                       Y_train.reshape(50000, ), \
                                       X_test.reshape(10000, 3072), \
                                       Y_test.reshape(10000, )
    net = Two_layers_dense_net(input_dim=3072, hidden_dim=512, num_classes=10, reg=0)
    X_train, Y_train = X_train[:2000, :] / 255 - 0.5, Y_train[:2000]
    X_test, Y_test = X_test[:500, :] / 255 - 0.5, Y_test[:500]
    data = {}
    data['X_train'], data['y_train'], data['X_test'], data['y_test'] = X_train, Y_train,\
                                                                       X_test, Y_test
    S = Solver(net, data,
               update_rule='Adam',
               optim_config={},
               lr=1e-2,
               print_every=10,
               check_point=check_point)
    S.train()
