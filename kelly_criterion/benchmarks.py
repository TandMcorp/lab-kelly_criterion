from timeit import default_timer as timer

import pandas as pd
from tqdm import tqdm


def run_estimator_benchmark(
    estimator, *, 
    n_samples=10,
    sample_sizes=None,
    max_sample_size_order=18,
):
    if sample_sizes is None:
        sample_sizes = [2**i for i in range(1, max_sample_size_order)]
        
    data = []
    for sample_size in tqdm(sample_sizes):
        for i in range(n_samples):
            t1 = timer()
            estimate = estimator(sample_size)
            t2 = timer()
            data.append({
                "sample_size": sample_size,
                "x": estimate.x,
                "f": estimate.f,
                "time": t2 - t1
            })
    
    return pd.DataFrame(data)