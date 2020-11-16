"""Test the code in models.py
"""

import diffeqinf
import numpy as np
import unittest


class TestNumericalForwardModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        y0 = np.array([1.0, 0.0])
        cls.model = diffeqinf.NumericalForwardModel(y0, 'RK45')

    def test_set_step_size(self):
        self.model.set_step_size(0.001)
        self.assertEqual(self.model.step_size, 0.001)

    def test_set_tolerance(self):
        self.model.set_tolerance(0.0001)
        self.assertEqual(self.model.tolerance, 0.0001)

    def test_n_parameters(self):
        with self.assertRaises(NotImplementedError):
            self.model.n_parameters()

    def test_simulate(self):
        with self.assertRaises(NotImplementedError):
            self.model.simulate([1, 2], [0, 1, 2])


class TestDampedOscillator(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.y0 = np.array([0.0, 0.0])

        def stimulus(t):
            return (1.0 * (t < 50)) + (0.0 * (t >= 50))

        cls.stimulus = stimulus
        cls.params = [1.0, 0.2, 1.0]
        cls.times = np.linspace(0, 100, 1000)

    def test_n_parameters(self):
        m = diffeqinf.DampedOscillator(self.stimulus, self.y0, 'RK45')
        self.assertEqual(m.n_parameters(), 3)

    def test_simulate(self):
        m = diffeqinf.DampedOscillator(self.stimulus, self.y0, 'RK45')
        m.set_tolerance(1e-5)
        y = m.simulate(self.params, self.times)
        self.assertEqual(len(y), len(self.times))

        m = diffeqinf.DampedOscillator(
            self.stimulus, self.y0, diffeqinf.ForwardEuler)
        m.set_step_size(0.001)
        y = m.simulate(self.params, self.times)
        self.assertEqual(len(y), len(self.times))
