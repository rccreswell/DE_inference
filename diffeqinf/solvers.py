"""Algorithms for solving differential equations.
"""

import numpy as np
import scipy.integrate
import scipy.interpolate
import warnings


class FDDenseOutput(scipy.integrate.DenseOutput):
    def __init__(self, t_old, t, y_old):
        super(FDDenseOutput, self).__init__(t_old, t)
        self.y_old = y_old

    def _call_impl(self, t):
        return self.y_old[0]
        # TODO: check/fix this


class ForwardEuler(scipy.integrate.OdeSolver):
    def __init__(self,
                 fun,
                 t0,
                 y0,
                 t_bound,
                 step_size=0.001,
                 vectorized=False,
                 **extraneous):

        if extraneous:
            warnings.warn('The following arguments have no effect for '
                          'Forward Euler solver: {}.'
                          .format(', '.join('`{}`'.format(x)
                                            for x in extraneous)))

        super(ForwardEuler, self).__init__(
            fun,
            t0,
            y0,
            t_bound,
            vectorized,
            support_complex=True)

        self.fixed_step_size = step_size

    def _step_impl(self):
        self.y_old = self.y.copy()

        self.y += self.fun(self.t, self.y) * self.fixed_step_size
        self.t += self.fixed_step_size

        if np.any(np.isinf(self.y)):
            return False, 'infinite y'

        return True, None

    def _dense_output_impl(self):
        return FDDenseOutput(self.t_old, self.t, self.y_old)
