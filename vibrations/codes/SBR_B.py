import numpy as np
import math


def hvf_weight(f_domain):
    """
    Scale frequency domain down to important frequencies.
    
    :param f_domain: (array) Frequency domain
    :return |Hv(f)|:
    """
    return 1 / np.sqrt(1 + (5.6 / f_domain)**2)


def v_eff_array(v, t):
    """
    Determine v_eff (voortschrijdende effectieve waarde) This is a sort of RMS-function.
    
    Explanation:
    integral from 0 to t.
    
    v_eff = sqrt( 1 /Ts ∫ g(ξ) v^2 (t - ξ) dξ
    
    at t;i
    
    |-----------------------|
    0                       t;i
    
    |-----------------------|
    ξ = t;i     <--      ξ = 0
    
    |-----------------------|
    v^2;0                  v^2;i
    
    :param v: (array) Frequency weighted signal transformed into time domain.
    :param t: (array)
    """
    Ts = 0.125
    v_eff = np.zeros(t.size)
    dt = t[1] - t[0]

    v_2 = v ** 2
    for i in range(t.size - 1):
        g_xi = np.exp(-t[:i + 1][::-1] / Ts)
        v_eff[i] = math.sqrt(1 / Ts * np.trapz(g_xi * v_2[:i + 1], dx=dt))

    return v_eff

