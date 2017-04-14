import numpy as np


def euler(t, f, initial=(0, 0)):
    """
    Eulers nummerical method.

    Computes dy and adds it to previous y-value.
    :param t: (list/ array) Time values of function f.
    :param f: (function) y'.
    :param initial:(tpl) Initial values.
    :return: (list/ array) y values.
    """
    # step size
    h = t[1] - t[0]
    y = np.zeros((t.size, ))

    t[0], y[0] = initial

    for i in range(t.size - 1):
        y[i + 1] = y[i] + h * f(t[i], y[i])

    return y


def runga_kutta_4(t, f, initial=(0, 0)):
    """
    Runga Kutta nummerical method.

    Computes dy and adds it to previous y-value.
    :param t: (list/ array) Time values of function f.
    :param f: (function)
    :param initial:(tpl) Initial values
    :return: (list/ array) y values.
    """
    # step size
    h = t[1] - t[0]
    y = np.zeros((t.size, ))

    t[0], y[0] = initial

    for i in range(t.size - 1):
        k1 = h * f(t[i], y[i])
        k2 = h * f(t[i] + h * 0.5, y[i] + 0.5 * k1)
        k3 = h * f(t[i] + h * 0.5, y[i] + 0.5 * k2)
        k4 = h * f(t[i + 1], y[i] + k3)
        y[i + 1] = y[i] + (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return y


