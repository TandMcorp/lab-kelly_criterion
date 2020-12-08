import ipympl
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits import mplot3d

EPSILON = 1e-8


def plot_3d_func(func, *, domain_lower=EPSILON, domain_upper=1-EPSILON, num_points=30):
    X, Y = np.meshgrid(
        np.linspace(domain_lower, domain_upper, num=num_points),
        np.linspace(domain_lower, domain_upper, num=num_points)
    )

    def bounded_func(x, y):
        if x + y > domain_upper:
            return np.nan
        else:
            return func(x, y)

    Z = np.vectorize(bounded_func)(X, Y)
    Z = np.nan_to_num(Z, nan=np.nanmin(Z))

    plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    return
