import numpy as np

"""
See: http://connor-johnson.com/2014/02/21/numerical-solutions-to-odes/

Steps divide the space between a and b

|__________________________________|
a                                   b
"""


def euler(a, b, n, f, initial=(0, 0)):
    """
    Eulers nummerical differential method.

    Computes dy and adds it to previous y-value.
    :param a: (flt) Start of the sequence.
    :param b: (flt) End of the sequence.
    :param n: (int) Number of steps
    :param f: (function)
    :param initial:(tpl) Initial values
    :return: (tpl) x and y values.
    """
    # step size
    h = (b - a) / n

    # mesh
    x = np.arange(a, b + h, h)
    y = np.zeros((n + 1, ))

    x[0], y[0] = initial

    for i in range(n):
        y[i + 1] = y[i] + h * f(x[i], y[i])

    return x, y


def runga_kutta_4(a, b, n, f, initial=(0, 0)):
    """
    Runga Kutta nummerical differential method.

    Computes dy and adds it to previous y-value.
    :param a: (flt) Start of the sequence.
    :param b: (flt) End of the sequence.
    :param n: (int) Number of steps
    :param f: (function)
    :param initial:(tpl) Initial values
    :return: (tpl) x and y values.
    """
    # step size
    h = (b - a) / n

    # mesh
    x = np.arange(a, b + h, h)
    y = np.zeros((n + 1, ))

    x[0], y[0] = initial

    for i in range(n):
        k1 = h * f(x[i], y[i])
        k2 = h * f(x[i] + h * 0.5, y[i] + 0.5 * k1)
        k3 = h * f(x[i] + h * 0.5, y[i] + 0.5 * k2)
        k4 = h * f(x[i + 1], y[i] + k3)
        y[i + 1] = y[i] + (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return x, y



