import os
import cProfile
from hallgerd.core import Sequential
from hallgerd.layers import Dense
import numpy as np


os.environ['PYOPENCL_CTX'] = '0'
os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'


def xor(x1, x2):
    if 1 / (x1 * x1) + 1 / (x2 * x2) > 4:
        #     if x1*x1 + x2*x2 < 3:
        return 1
    return 0


if __name__ == '__main__':

    vxor = np.vectorize(xor)
    X = np.random.randn(10000, 2, )
    y = vxor(X[:, 0], X[:, 1])
    X = X.T
    y = y.reshape((1, -1))

    model = Sequential(lr=1e-2, batch_size=256, epochs=64)
    model.add(Dense(2, 4))
    model.add(Dense(4, 1))

    cProfile.run('model.fit(X, y)', filename='profile.log')
