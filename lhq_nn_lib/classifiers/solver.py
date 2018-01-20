import numpy as np
import pickle
from classifiers import optim

class Solver():
    def __init__(self, model, data, **kwargs):
        """
        including training function and test function
        :param model: model of the network
        :param data: input data
        :param kwargs:
        - update_rule: A string of the function name in layers/optim.py
        - optim_config: A dictionary of the hyper parameters in the update_rule you
                        have chosen
        - lr_decay: lr_decay rate
        - lr: Learning rate
        - batch_size: Size of minibatches
        - num_epochs: The number of epochs in training function
        - print_every: Training loss print in every print_every step
        - num_train_sample: Number of training sample to compute accuracy
        - num_test_sample: Number of testing sample to compute accuracy
        - checkpoint_name: To save the params of model in this file
        - save_iter: after save_iter iteration, save the model
        """
        self.model = model
        self.X_train = data['X_train']
        self.y_train = data['y_train']
        self.X_test = data['X_test']
        self.y_test = data['y_test']

        self.update_rule = kwargs.pop('update_rule', 'sgd')
        self.optim_config = kwargs.pop('optim_config', {})
        self.lr_decay = kwargs.pop('lr_decay', 1.0)
        self.lr = kwargs.pop('lr', 1e-3)
        self.batch_size = kwargs.pop('batch_size', 64)
        self.num_epochs = kwargs.pop('num_epochs', 10)
        self.print_every = kwargs.pop('print_every', 10)
        self.num_train_sample = kwargs.pop('num_train_sample', 500)
        self.num_test_sample = kwargs.pop('num_test_sample', 100)
        self.checkpoint_name = kwargs.pop('check_point', 'model')
        self.save_iter = kwargs.pop('save_iter', 100)

        if len(kwargs) > 0:
            extra = ', '.join('"%s"' % k for k in list(kwargs.keys()))
            raise ValueError('Unrecognized arguments %s' % extra)
        if not hasattr(optim, self.update_rule):
            raise ValueError('Invalid update_rule "%s"' % self.update_rule)
        self.update_rule = getattr(optim, self.update_rule)
        self._reset()

    def _reset(self):
        self.epoch = 0
        self.best_val = 0
        self.best_params = {}
        self.loss_history = []
        self.train_acc_history = []
        self.test_acc_history = []

        self.optim_configs = {}
        for param in self.model.params:
            d = {k: v for k, v in self.optim_config.items()}
            self.optim_configs[param] = d

    def _step(self):
        num_train = self.X_train.shape[0]
        batch_indices = np.random.choice(num_train, self.batch_size)
        X_batch = self.X_train[batch_indices]
        y_batch = self.y_train[batch_indices]

        output = self.model.forward(X_batch)
        loss, d_loss = self.model.compute_loss(output, y_batch)
        dx, grads = self.model.backward(d_loss)

        self.loss_history.append(loss)

        for p, w in self.model.params.items():
            dw = grads[p]
            config = self.optim_configs[p]
            next_w, next_config = self.update_rule(w, dw, self.lr, config)
            self.model.params[p] = next_w
            self.optim_configs[p] = next_config

    def _save_checkpoint(self):
        checkpoint = {
            'model': self.model,
            'update_rule': self.update_rule,
            'lr_decay': self.lr_decay,
            'lr': self.lr,
            'optim_config': self.optim_config,
            'batch_size': self.batch_size,
            'num_train_sample': self.num_train_sample,
            'num_test_sample': self.num_test_sample,
            'epoch': self.epoch,
            'loss_history': self.loss_history,
            'train_acc_history': self.train_acc_history,
            'test_acc_history': self.test_acc_history
        }
        filename = '%s_epoch_%d.pkl' % (self.checkpoint_name, self.epoch)
        with open(filename, 'wb') as file:
            pickle.dump(checkpoint, file)

    def check_accuracy(self, X, y, num_sample=None, batch_size=64):
        N = X.shape[0]
        if num_sample is not None and N > num_sample:
            indices = np.random.choice(N, num_sample)
            N = num_sample
            X = X[indices]
            y = y[indices]

        num_batch = N//batch_size
        if N % batch_size != 0:
            num_batch += 1
        y_pred = []
        for i in range(num_batch):
            start = i * batch_size
            end = start + batch_size
            scores = self.model.forward(X[start: end])
            y_pred.append(np.argmax(scores, axis=1))
        y_pred = np.hstack(y_pred)
        acc = np.mean(y_pred == y)
        return acc

    def train(self):
        num_train = self.X_train.shape[0]
        iterations_per_epoch = max(num_train // self.batch_size, 1)
        num_iterations = self.num_epochs * iterations_per_epoch

        for i in range(num_iterations):
            self._step()
            if i % self.print_every == 0:
                print('(Iteration %d / %d) loss: %f' % (
                       i + 1, num_iterations, self.loss_history[-1]))
            if i % self.save_iter == 0:
                self._save_checkpoint()
                print('Check point saved')
            epoch_end = (i + 1) % iterations_per_epoch
            if epoch_end:
                self.epoch += 1
                self.lr *= self.lr_decay

            first_it = (i == 0)
            last_it = (i == num_iterations-1)
            if first_it or last_it or epoch_end:
                train_acc = self.check_accuracy(self.X_train,
                                                self.y_train, self.num_train_sample, self.batch_size)
                test_acc = self.check_accuracy(self.X_test, self.y_test,
                                               self.num_test_sample, self.batch_size)
                self.train_acc_history.append(train_acc)
                self.test_acc_history.append(test_acc)
                # self._save_checkpoint()
                print('(Epoch %d / %d) train acc: %f; val_acc: %f' % (
                    self.epoch, self.num_epochs, train_acc, test_acc))
                if test_acc > self.best_val:
                    self.best_val = test_acc
                    self.best_params = {}
                    for k, v in self.model.params.items():
                        self.best_params[k] = v.copy()
        self.model.params = self.best_params

