import numpy as np
import unittest
from ode.nummerical_ode import euler, runga_kutta_4
from ode.vibration import finite_difference_method, det_damping, runga_kutta_vibrations, scipy_ode_vibrations, \
    det_response_spectrum


class ODE(unittest.TestCase):
    def test_ode(self):
        # ordinary differential equation
        def f(x, y):
            return y - x ** 2 + 1
        t = np.arange(0, 2.2, 0.2)
        y_val_euler = euler(t, f, (0, 0.5))
        y_val_rk = runga_kutta_4(t, f, (0, 0.5))

        y_euler_sol = [0.5, 0.80000000000000004, 1.1520000000000001, 1.5504000000000002, 1.9884800000000002, 2.4581760000000004, 2.9498112000000005, 3.4517734400000006, 3.9501281280000007, 4.4281537536000011, 4.8657845043200014]
        y_rk_sol = [0.5, 0.82929333333333344, 1.2140762106666667, 1.6489220170416001, 2.1272026849479437, 2.6408226927287517, 3.1798941702322305, 3.7323400728549796, 4.283409498318405, 4.8150856945794329, 5.3053630006926529]

        for i in range(len(t)):
            self.assertEqual(y_val_euler[i], y_euler_sol[i])
            self.assertEqual(y_val_rk[i], y_rk_sol[i])

    def test_vibrations(self):
        n = 200
        t = np.linspace(0, 1, n)

        c = 0
        force = np.zeros(n)
        with open("force.txt") as f:
            for l in f:
                force[c] = float(l)
                c += 1

        k = 42940
        m = 200
        c = det_damping(k, m, 0.02)
        yf = finite_difference_method(t, 0, m, c, k, force)
        y_rk = runga_kutta_vibrations(t, 0, 0, m, c, k, force)[0]
        y_ode = scipy_ode_vibrations(t, 0, 0, m, c, k, force)[0]

        for i in range(yf.size):
            self.assertAlmostEqual(yf[i], y_rk[i], 2)
            self.assertAlmostEqual(yf[i], y_ode[i], 2)

