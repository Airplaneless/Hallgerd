import numpy as np
import logging


def mse(yt, yp):
    logging.debug('CPU::compute MSE')
    return np.linalg.norm(yt - yp)


def mse_delta(yt, yp):
    logging.debug('CPU::compute dMSE')
    return 2 * (yt - yp)


def softmax(x):
    logging.debug('CPU::compute softmax')
    exps = np.exp(x - np.max(x))
    return exps / np.sum(exps, axis=0)


def cross_entropy(yt, yp):
    logging.debug('CPU::compute cross_entropy')
    p = softmax(yp)
    loss = -np.sum(yt * p)
    return loss


def cross_entropy_delta(yt, yp):
    logging.debug('CPU::compute dcross_entropy')
    p = softmax(yp)
    return yt - yp