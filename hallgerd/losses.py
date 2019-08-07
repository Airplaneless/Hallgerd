import numpy as np


def mse(yt, yp):
    return np.linalg.norm(yt - yp)


def mse_delta(yt, yp):
    return 2 * (yt - yp)


def softmax(x):
    exps = np.exp(x - np.max(x))
    return exps / np.sum(exps, axis=0)


def cross_entropy(yt, yp):
    loss = -np.sum(yt * yp)
    return loss


def cross_entropy_delta(yt, yp):
    return yt - yp