import numpy as np
import math
from scipy.integrate import odeint


def scipy_ode_vibrations(t, u0, v0, m, c, k, force):
    """
    Uses the the scipy ode solver to solve a single mass spring system.

    mx'' + cx' + kx = F

    :param t: (list/ array) Time.
    :param u0: (flt)u at t[0]
    :param v0: (flt) v at t[0].
    :param force: (list/ array) Force acting on the system.
    :param c: (flt) Damping.
    :param k: (flt) Spring stiffness.
    :param m:(flt) Mass.
    :return: (tpl) (displacement u, velocity v)
    """

    def func(Y, t, force, c, k, m, time_arr):
        """
        :param Y: (tpl) (displacement u, velocity v)
        :param t: (list/ array) Time.
        :param force: (list/ array) Force acting on the system.
        :param c: (flt) Damping.
        :param k: (flt) Spring stiffness.
        :param m:(flt) Mass.
        :param time_arr: (list/ array) Time.
        :return: (tpl) (velocity v, displacement u)
        """
        return Y[1], (np.interp(t, time_arr, force) - c * Y[1] - k * Y[0]) / m

    s = odeint(func, [u0, v0], t, args=(force, c, k, m, t))

    return s[:, 0], s[:, 1]


def runga_kutta_vibrations(t, u0, v0, m, c, k, force):
    """
    Solve a single mass spring system with 4th order Runga Kutta.

    mu'' + cu' + ku = F

    rewrite x:
    u' = v

    makes:
    mv' + cv + ku = F
    v' = (F - cv - ku) / m

    We now have got two ode's

    x' = v
    v' = (F - cv - ku) / m

    :param t: (list/ array)
    :param u0: (flt)u at t[0]
    :param v0: (flt) v at t[0].
    :param m:(flt) Mass.
    :param c: (flt) Damping.
    :param k: (flt) Spring stiffness.
    :param force: (list/ array) Force acting on the system.
    :return: (tpl) (displacement u, velocity v)
    """

    u = np.zeros(t.shape)
    v = np.zeros(t.shape)
    u[0] = u0
    v[0] = v0
    dt = t[1] - t[0]

    def func(u, V, force):
        return (force - c * V - k * u) / m

    for i in range(t.size - 1):
        # F at time step t / 2
        f_t_05 = (force[i + 1] - force[i]) / 2 + force[i]

        u1 = u[i]
        v1 = v[i]
        a1 = func(u1, v1, force[i])
        u2 = u[i] + v1 * dt / 2
        v2 = v[i] + a1 * dt / 2
        a2 = func(u2, v2, f_t_05)
        u3 = u[i] + v2 * dt / 2
        v3 = v[i] + a2 * dt / 2
        a3 = func(u3, v3, f_t_05)
        u4 = u[i] + v3 * dt
        v4 = v[i] + a3 * dt
        a4 = func(u4, v4, force[i + 1])
        u[i + 1] = u[i] + dt / 6 * (v1 + 2 * v2 + 2 * v3 + v4)
        v[i + 1] = v[i] + dt / 6 * (a1 + 2 * a2 + 2 * a3 + a4)

    return u, v


def finite_difference_method(t, u0, m, c, k, force):
    """
    Determine the reaction of a one mass spring system to an outside force.
    See: http://folk.uio.no/hpl/scripting/bumpy-dir/sphinx-rootdir/_build/html/vibcase.html

    :param t: (list/ array) With time values.
    :param u0: (flt) Boundary condition for the displacement u at t.
    :param m: (flt) Mass.
    :param c: (flt) Damping.
    :param k: (flt) Spring stiffness.
    :param force: (list/ array) Force acting on the system.
    :return: (array) Displacements.
    """

    dt = t[1] - t[0]
    u = np.zeros(t.size)
    u[0] = u0
    u[1] = u[0] + dt**2 / (2 * m) * (k * u0 - force[0])

    for i in range(1, t.size - 1):
        u[i + 1] = ((c / 2 * dt - m) * u[i - 1] + 2 * m * u[i] - dt**2 * (k * u[i] - force[i])) \
                    / (c / 2 * dt + m)
    return u


def det_damping(k, m, zeta):
    """
    Determine the damping.

    :param k: (flt) Spring stiffness.
    :param m: (flt) Mass.
    :param zeta: (flt) Damping ratio ζ = c / ckr.
    :return: (flt) Damping c.
    """
    return zeta * det_critical_damping(k, m)


def det_damping_ratio(k, m, c):
    """
    :param k: (flt) Spring stiffness.
    :param m: (flt) Mass.
    :param c: (flt) Damping.
    :return: (flt) ζ damping ratio.
    """
    return c / det_critical_damping(k, m)


def det_critical_damping(k, m):
    return 2 * m * math.sqrt(k / m)


def det_natural_frequency(k, m):
    return math.sqrt(k / m)


def det_period(k, m):
    """
    :param k: (flt) Spring stiffness.
    :param m: (flt) Mass.
    :return: (flt) period T.
    """
    return 2 * math.pi * math.sqrt(m / k)


def det_spring_stiffness(m, T):
    """
    :param m: (flt) Mass.
    :param T: (flt) Period.
    :return: (flt) Spring stiffness k.
    """
    return m / (T / (2 * math.pi))**2


def det_response_spectrum(func, period, t, u0, v0, m, damping_ratio, force):
    """
    Only usable with Runga Kutta and Scipy solver.

    :param func: (vibrations function).
    :param period: (array) factor by which to multiply k. This way the natural frequency will be modified.
    :param t: (list/ array) With time values.
    :param u0: (flt)u at t[0]
    :param v0: (flt) v at t[0].
    :param m: (flt) Mass.
    :param damping_ratio: (flt) ζ damping ratio. Example: 0.05.
    :param force: (list/ array) Force acting on the system.
    :return: (array) Containing the maximum absolute values per multiplied factor. This thus resembles the maximum
                    amplitude per natural frequency.
    """
    u = np.zeros(period.size)
    v = np.zeros(period.size)
    a = np.zeros(period.size)

    for i in range(period.size):
        k = det_spring_stiffness(m, period[i])
        c = det_damping(k, m, damping_ratio)
        sol = func(t, u0, v0, m, c, k, force)
        u[i] = np.max(np.absolute(sol[0]))
        v[i] = np.max(np.absolute(sol[1]))
        a[i] = np.max(np.absolute(np.diff(sol[1])))

    return u, v, a

