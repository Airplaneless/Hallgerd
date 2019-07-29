import logging
import numpy as np
import pyopencl as cl
from tqdm import tqdm

from hallgerd.cl import MAT_CL_KERNELS
from hallgerd.layers import Dense
from hallgerd.losses import *


SUPPORTED_LOSSES = ['mse', 'cross_entropy']


class Sequential:

    def __init__(self, lr=1e-3, batch_size=1, epochs=1, loss='mse', verbose=True):
        self.lr = lr
        self.bs = batch_size
        assert loss in SUPPORTED_LOSSES
        self.loss = loss
        self.epochs = epochs
        self.history = {}
        self.verbose = verbose
        self.layers = list()
        platforms = cl.get_platforms()
        if not platforms:
            raise RuntimeError('No OpenCL platforms')
        else:
            p = platforms[0]
            devices = p.get_devices()
            if not devices:
                raise RuntimeError('No OpenCL devices')
            logging.info('utilizing  {}'.format(devices[0].name))
        logging.debug('creating context')
        self.ctx = cl.create_some_context()
        self.queue = cl.CommandQueue(self.ctx)
        self.prg = cl.Program(self.ctx, MAT_CL_KERNELS).build()

    def add(self, layer):
        layer.__connect_context__(self.ctx, self.queue, self.prg)
        self.layers.append(layer)

    def __call__(self, x):
        logging.debug('evaluate for {}'.format(x.shape))
        _batches = x.shape[1]
        x = x.copy().astype(np.float64)
        logging.debug('CL::start copy to device')
        x_cl = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=x)
        logging.debug('CL::finished copy to device')
        for layer in self.layers:
            x_cl = layer(x_cl, batches=_batches)
        out_np = np.empty((self.layers[-1].out_shape, _batches), dtype=np.float64)
        logging.debug('CL::start copy from device')
        cl.enqueue_copy(self.queue, out_np, x_cl)
        logging.debug('CL::finished copy from device')
        return out_np

    def weights2cpu(self):
        logging.info('CL::copy weights to host')
        for layer in self.layers:
            layer.__weight2cpu__()
        return True

    def backprop(self, y):
        logging.debug('start backpropagation')
        y = y.copy().astype(np.float64)
        logging.debug('CL::start copy from device')
        yp = np.empty((self.layers[-1].out_shape, y.shape[1]), dtype=np.float64)
        cl.enqueue_copy(self.queue, yp, self.layers[-1].output_cl)
        logging.debug('CL::finished copy from device')
        if self.loss == 'mse':
            error = mse_delta(y, yp)
        if self.loss == 'cross_entropy':
            error = cross_entropy_delta(y, yp)
        logging.debug('CL::start copy to device')
        error_cl = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=error)
        logging.debug('CL::finished copy to device')
        for layer in reversed(self.layers):
            error_cl = layer.backprop(error_cl, self.lr)
        error_cl.release()
        return True

    def fit(self, X, y):
        assert X.shape[1] == y.shape[1]
        logging.info('fitting on {} data'.format(X.shape))
        num_batches = np.ceil(X.shape[1] / self.bs)
        self.history['loss'] = list()
        for _ in tqdm(range(self.epochs), disable=not self.verbose):
            for x, yt in zip(np.array_split(X.T, num_batches), np.array_split(y.T, num_batches)):
                _ = self.__call__(x.T)
                self.backprop(yt.T)
            if self.loss == 'mse':
                loss = mse(y, self.__call__(X))
            if self.loss == 'cross_entropy':
                loss = cross_entropy(y, self.__call__(X))
            logging.info('train loss: {}'.format(loss))
            self.history['loss'].append(loss)


if __name__ == '__main__':

    X = np.random.random((784, 1000))
    y = np.random.randint(0, 10, (10, 1000))

    model = Sequential(lr=1e-1, batch_size=512, epochs=1, loss='cross_entropy')
    model.add(Dense(784, 512, activation='relu'))
    model.add(Dense(512, 512, activation='relu'))
    model.add(Dense(512, 10, activation='softmax'))
    model.fit(X, y)

    # X = np.random.random((2, 30)).astype(np.float64)
    # X = X.copy()
    print(model(X[:, 0:5]))
    # print(model(X[:, 0:20]))
    # print(model(x))
