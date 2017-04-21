import numpy as np


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


def FFT(y, t):
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
