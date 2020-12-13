import numpy as np

from .estimators import make_univariate_saa_estimator


def estimate_univariate_kelly_strategy(sampler, *, sample_size=1000000):
    estimate = make_univariate_saa_estimator(
        sampler,
        func=univariate_log_return,
        func_to_minimize=lambda x, samples: -univariate_log_return(x, samples)
    )(sample_size)
    return estimate["x"]


def univariate_log_return(x, samples):
    return np.sum(np.log(1 + x * samples))/len(samples)


def grad_univariate_log_return(x, samples):
    return np.sum(samples / (1 + x * samples))/len(samples)


def log_return(x, samples):
    return np.sum(np.log(1 + np.matmul(samples, x)))/len(samples)


def grad_log_return(x, samples):
    return np.sum(samples / (1 + np.matmul(samples, x)))/len(samples)
