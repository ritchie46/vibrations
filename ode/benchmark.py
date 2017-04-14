import numpy as np
from ode.vibration import forced_vibrations, runga_kutta_vibrations, scipy_ode_vibrations
import time


func = [runga_kutta_vibrations, forced_vibrations, scipy_ode_vibrations]
m = 100
c = 100
k = 100

t = np.arange(0, 10000, 1)
force = np.sin(t) * 100

print("Big steps")
c = 0
for f in func:
    if c == 1:
        t0 = time.time()
        f(t, 3, m, c, k, force)
        print(time.time() - t0)
    else:
        t0 = time.time()
        f(t, 3, 0, m, c, k, force)
        print(time.time() - t0)
    c += 1


t = np.arange(0, 1, 1 / 10000)
force = np.sin(t) * 100
print("Small steps")
c = 0
for f in func:
    if c == 1:
        t0 = time.time()
        f(t, 3, m, c, k, force)
        print(time.time() - t0)
    else:
        t0 = time.time()
        f(t, 3, 0, m, c, k, force)
        print(time.time() - t0)
    c += 1