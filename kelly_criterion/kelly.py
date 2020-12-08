import numpy as np


def univariate_log_return(x, samples):
    return np.sum(np.log(1 + x * samples))/len(samples)


def grad_univariate_log_return(x, samples):
    return np.sum(samples / (1 + x * samples))/len(samples)


def log_return(x, samples):
    return np.sum(np.log(1 + np.matmul(samples, x)))/len(samples)


def grad_log_return(x, samples):
    return np.sum(samples / (1 + np.matmul(samples, x)))/len(samples)
