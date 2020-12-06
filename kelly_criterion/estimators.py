import pandas as pd
from scipy.optimize import minimize_scalar


def make_univariate_saa_estimator(sampler, func, *, func_to_minimize=None):
    if func_to_minimize is None:
        func_to_minimize = func

    def estimator(sample_size):
        samples = sampler(sample_size)
        res = minimize_scalar(func_to_minimize, args=(samples,), method="bounded", bounds=(0, 1))
        if not res.success:
            return None
        return {
            "x": res.x,
            "f": func(res.x, samples),
        }

    return estimator


def make_univariate_sgd_estimator(
    sampler, grad, *,
    batch_size=1,
    x0,
    step_size_multiplier=1,
    include_trace=False
):

    def estimator(sample_size):
        trace = []
        x = x0
        for i in range(1, sample_size+1):
            samples = sampler(batch_size)
            grad_i = grad(x, samples)
            step_size = step_size_multiplier / i
            x += step_size * grad_i
            if x < 0:
                x = 0
            elif x > 1:
                x = 1
            if include_trace:
                trace.append((i, x))

        out = {"x": x}
        if include_trace:
            out["trace"] = pd.DataFrame(trace, columns=["i", "x"])

        return out

    return estimator
