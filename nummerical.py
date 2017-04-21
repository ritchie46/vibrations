import numpy as np
from scipy import integrate


def integrate_array(y, x):
    """
    integrate array y * dx
    """
    y_int = np.zeros(x.size)
    for i in range(x.size - 1):
        y_int[i + 1] = y_int[i] + y[i + 1] * (x[i + 1] - x[i])
    return y_int
