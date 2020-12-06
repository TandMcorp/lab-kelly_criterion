from dataclasses import dataclass
from typing import Optional

import altair as alt
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import bernoulli
from tqdm import tqdm



@dataclass
class Estimate:
    x: float
    f: float


def make_saa_estimator(sampler, func, *, func_to_minimize=None):
    if func_to_minimize is None:
        func_to_minimize = func

    def estimator(sample_size) -> Optional[Estimate]:
        samples = sampler(sample_size)
        res = minimize_scalar(func_to_minimize, args=(samples,), method="bounded", bounds=(0, 1))
        if not res.success:
            return None
        return Estimate(
            x=res.x,
            f=func(res.x, samples),
        )

    return estimator
