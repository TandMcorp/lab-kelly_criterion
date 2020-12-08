from scipy.stats import bernoulli
from scipy.stats import uniform


def make_bernoulli_return_sampler(p):
    def sampler(size):
        return 2 * bernoulli.rvs(p, size=size)-1

    return sampler

def make_uniform_return_sampler(upper, *, lower=-1):
    def sampler(size):
        return uniform.rvs(loc=lower, scale=upper-lower, size=size)

    return sampler
