import numpy as np


def univariate_log_return(x, samples):
    return np.sum(np.log(1 + x * samples))/len(samples)

