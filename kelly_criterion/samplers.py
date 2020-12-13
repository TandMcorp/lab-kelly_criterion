import numpy as np
from scipy.stats import bernoulli
from scipy.stats import lognorm
from scipy.stats import uniform


def make_bernoulli_return_sampler(p, *, n=1):
    def sampler(size):
        return 2 * bernoulli.rvs(p, size=(size, n))-1

    return sampler

def make_uniform_return_sampler(upper, *, lower=-1, n=1):
    def sampler(size):
        return uniform.rvs(loc=lower, scale=upper-lower, size=(size, n))

    return sampler

def make_lognormal_return_sampler(mu, sigma, *, n=1):
    def sampler(size):
        return lognorm.rvs(sigma, scale=np.exp(mu), size=(size, n)) - 1

    return sampler
