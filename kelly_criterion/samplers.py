from scipy.stats import bernoulli


def make_bernoulli_return_sampler(p):
    def sampler(size):
        return 2 * bernoulli.rvs(p, size=size)-1

    return sampler
