import numpy as np
import math


def integrate_array(y, t):
    """
    integrate array y * dx
    """
    y_int = np.zeros(t.size)
    for i in range(t.size - 1):
        y_int[i + 1] = y_int[i] + y[i + 1] * (t[i + 1] - t[i])
    return y_int


def differentiate_array(y):
    return np.diff(y)


def fft(y, t):
    """
    Return the FFT (Fast Fourier Transform) for a real valued signal. 
    Because of symmetry only half of the FFT is returned.
    
    :param y: (array) Amplitude values of time spectrum.
    :param t: (array) Time values of time spectrum.
    :return: (tpl) Containing the frequency values and the amplitude values of the frequency spectrum.
    """
    T = t[2] - t[1]
    N = t.size // 2
    # amplitude
    yf = np.abs(np.fft.fft(y)[:N])

    # frequency
    f = np.linspace(0, 1 / (2 * T), N)
    return f, yf


def rms(y):
    """
    :param y: (array)
    :return: (flt) Root mean squared of y.
    """
    return np.sqrt(np.mean(y**2))


def rms_array(y):
    """
    :param y: (array)
    :return: (array) Root mean squared of y for every y_i.
    """
    y_rms = np.empty(y.size)
    y_rms[0] = y[0]

    for i in range(y.size - 1):
        y_rms[i + 1] = math.sqrt((y_rms[i]**2 * (i + 1) + y[i + 1]**2) / (i + 2))
    return y_rms

