"""Pints models looking at tolerance and step size.
"""

import pints
import pints.plot
import scipy
import scipy.integrate
import scipy.interpolate
import math


class NumericalForwardModel(pints.ForwardModel):
    def __init__(self, y0, solver, step_size=None, tolerance=None):
        """
        Parameters
        ----------
        y0 : np.ndarray
            Initial condition
        solver : str or OdeSolver
            Method for solving ODE. Can be a str from the scipy choices or
            another OdeSolver.
        step_size : float, optional
            Step size used in the solver
        tolerance : float, optional
            Tolerance used in the solver
        """
        self.y0 = y0
        self.solver = solver
        self.step_size = step_size
        self.tolerance = tolerance

    def set_step_size(self, step_size):
        self.step_size = step_size

    def set_tolerance(self, tolerance):
        self.tolerance = tolerance

    def n_parameters(self):
        raise NotImplementedError

    def simulate(self, parameters, times):
        raise NotImplementedError


class DampedOscillator(NumericalForwardModel):
    def __init__(self, stimulus, y0, solver, step_size=None, tolerance=None):
        """
        Parameters
        ----------
        stimulus : function
            Input stimulus as a function of time
        """
        super(DampedOscillator, self).__init__(
            y0, solver, step_size=step_size, tolerance=tolerance)

        self.stimulus = stimulus

    def n_parameters(self):
        return 3

    def simulate(self, parameters, times):
        k = parameters[0]
        c = parameters[1]
        m = parameters[2]
        w = math.sqrt(k/m)
        g = c / (2 * math.sqrt(m * k))

        def fun(t, y):
            # TODO: fix
            try:
                return [y[1],
                        self.stimulus(t) / m - 2*g*w*y[1] - w**2 * y[0]]
            except TypeError:
                return [y[1],
                        self.stimulus.__func__(t) / m - 2*g*w*y[1]
                        - w**2 * y[0]]

        t_range = (min(times), max(times))

        res = scipy.integrate.solve_ivp(
            fun,
            t_range,
            [0, 0],
            t_eval=times,
            method=self.solver,
            rtol=self.tolerance,
            step_size=self.step_size
        )
        y = res.y
        if y.ndim >= 2:
            y = res.y[0]

        return y
