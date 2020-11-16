"""Python functions for making the figures.
"""

import matplotlib.pyplot as plt
import numpy as np
import pints
import diffeqinf


def figure1():
    """Make an interactive figure for numerical error.
    """

    def stimulus(t):
        return (1 * (t < 50)) + (-100 * (t >= 50) & (t < 75)) + (1 * (t >= 75))

    # Generate data
    y0 = np.array([0.0, 0.0])
    m = diffeqinf.DampedOscillator(stimulus, y0, 'RK45')
    m.set_tolerance(1e-8)
    true_params = [1.0, 0.2, 1.0]
    times = np.linspace(0, 100, 500)
    y = m.simulate(true_params, times)
    y += np.random.normal(0, 0.01, len(times))

    # Forward Euler method
    m = diffeqinf.DampedOscillator(stimulus, y0, diffeqinf.ForwardEuler)
    m.set_step_size(0.01)
    problem = pints.SingleOutputProblem(m, times, y)
    likelihood = pints.GaussianLogLikelihood(problem)

    step_sizes = [0.2, 0.1, 0.01]
    true_params = [1.0, 0.2, 1.0, 0.01]

    diffeqinf.plot.plot_likelihoods(
        problem,
        likelihood,
        true_params,
        step_sizes=step_sizes,
        param_names=['k', 'c', 'm'])

    # RK45 method
    m = diffeqinf.DampedOscillator(stimulus, y0, 'RK45')
    m.set_tolerance(1e-2)

    problem = pints.SingleOutputProblem(m, times, y)
    likelihood = pints.GaussianLogLikelihood(problem)

    tolerances = [0.01, 0.0001, 0.000001]

    diffeqinf.plot.plot_likelihoods(
        problem,
        likelihood,
        true_params,
        tolerances=tolerances,
        param_names=['k', 'c', 'm'])


def figure2():
    """Make a figure for MCMC inference
    """
    num_mcmc_iters = 10000

    def stimulus(t):
        return (1 * (t < 50)) + (-100 * (t >= 50) & (t < 75)) + (1 * (t >= 75))

    # Generate data
    y0 = np.array([0.0, 0.0])
    m = diffeqinf.DampedOscillator(stimulus, y0, 'RK45')
    m.set_tolerance(1e-8)
    true_params = [1.0, 0.2, 1.0]
    times = np.linspace(0, 100, 500)
    y = m.simulate(true_params, times)
    y += np.random.normal(0, 0.01, len(times))

    # Run inference with correct model
    problem = pints.SingleOutputProblem(m, times, y)
    likelihood = pints.GaussianLogLikelihood(problem)
    prior = pints.UniformLogPrior([0]*4, [1e6]*4)
    posterior = pints.LogPosterior(likelihood, prior)

    x0 = [true_params + [0.01]] * 3

    mcmc = pints.MCMCController(posterior, 3, x0)
    mcmc.set_max_iterations(num_mcmc_iters)
    chains_correct = mcmc.run()

    # Run inference with incorrect model
    m.set_tolerance(1e-2)
    problem = pints.SingleOutputProblem(m, times, y)
    likelihood = pints.GaussianLogLikelihood(problem)
    prior = pints.UniformLogPrior([0]*4, [1e6]*4)
    posterior = pints.LogPosterior(likelihood, prior)

    mcmc = pints.MCMCController(posterior, 3, x0)
    mcmc.set_max_iterations(num_mcmc_iters)
    chains_incorrect = mcmc.run()

    # Plot MCMC chains
    pints.plot.trace(chains_incorrect)
    plt.show()

    # Plot posteriors
    diffeqinf.plot.plot_grouped_parameter_posteriors(
        [chains_correct[0, num_mcmc_iters//2:,:]],
        [chains_incorrect[0, num_mcmc_iters//2:,:]],
        [chains_incorrect[1, num_mcmc_iters//2:,:]],
        [chains_incorrect[2, num_mcmc_iters//2:,:]],
        true_model_parameters=true_params,
        method_names=['Correct', 'PoorTol_Chain1',
                      'PoorTol_Chain2', 'PoorTol_Chain3'],
        parameter_names=['k', 'c', 'm'],
        fname=None)
    plt.show()


if __name__ == '__main__':
    figure1()
    # figure2()
