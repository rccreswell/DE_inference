"""Test the code in solvers.py
"""

import diffeqinf
import numpy as np
import scipy.integrate
import unittest


class TestForwardEuler(unittest.TestCase):

    def test_solve(self):
        # Test that it solves a differential equation
        times = np.linspace(0, 10, 100)
        t_range = (min(times), max(times))

        def fun(t, y):
            return y * (1 - y)

        res = scipy.integrate.solve_ivp(
            fun,
            t_range,
            [0.5],
            t_eval=times,
            method=diffeqinf.ForwardEuler,
            step_size=0.0001
        )

        self.assertTrue(
            np.allclose(res.y, 1 / (1 + np.exp(-times)), rtol=1e-4))
