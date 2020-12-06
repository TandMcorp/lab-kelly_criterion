import altair as alt
import numpy as np
import pandas as pd
from tqdm import tqdm_notebook as tqdm

EPSILON = 1e-8


def plot_func(func, *, x_lower=EPSILON, x_upper=1-EPSILON, num_points=1000):
    xs = np.linspace(x_lower, x_upper, num=num_points)
    fs = np.array([func(x) for x in xs])
    
    alt.data_transformers.disable_max_rows()

    return (
        alt
        .Chart(
            pd.DataFrame({"x": xs, "func": fs})
        )
        .mark_line()
        .encode(
            x="x",
            y="func",
        )
    )


def plot_sampled_func(
    func, sampler, *,
    x_lower=EPSILON,
    x_upper=1-EPSILON,
    num_points=1000,
    num_func_samples=10, 
    across_sample_sizes=[10, 100, 1000]
):
    xgrid = list(np.linspace(x_lower, x_upper, num=num_points))

    sample_sizes = []
    instances = []
    xs = []
    fs = []
    for sample_size in tqdm(across_sample_sizes, desc="sample_size", leave=False):
        for i in tqdm(range(num_func_samples), desc="func_sample", leave=False):
            samples = sampler(sample_size)
            sample_sizes += [sample_size] * num_points
            instances += [i] * num_points
            xs += xgrid
            fs += [func(x, samples=samples) for x in xgrid]
    
    alt.data_transformers.disable_max_rows()

    return (
        alt
        .Chart(
            pd.DataFrame({"sample_size": sample_sizes, "instance": instances, "x": xs, "func": fs})
        )
        .mark_line()
        .encode(
            x="x",
            y="func",
            column="sample_size:O",
            detail="instance",
            opacity=alt.value(0.2),
        )
    )

def plot_benchmark_stats(stats, *, by="sample_size"):
    return (
        alt
        .Chart(
            stats.melt(id_vars=[by])
        )
        .mark_boxplot(median={
            "color": "black"
        })
        .encode(
            x=alt.X("sample_size", scale=alt.Scale(type="log")),
            y=alt.Y("value"),
            column="variable",
        )
        .resolve_scale(y='independent')
    )
