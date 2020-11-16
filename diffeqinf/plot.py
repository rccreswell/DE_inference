"""Plotting functions for ODE inference.
"""

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons
import numpy as np


def plot_likelihoods(problem,
                     likelihood,
                     true_params,
                     tolerances=None,
                     step_sizes=None,
                     param_names=None,
                     figsize=(10, 10)):
    """
    Parameters
    ----------
    problem : pints.SingleOutputProblem
        pints problem
    likelihood : pints.LogLikelihood
    step_sizes : list
        Set of step sizes to try
    """

    # Get generic parameter names if not provided
    if param_names is None:
        param_names = ['Param{}'.format(i)
                       for i in range(problem.n_parameters())]

    # Data times and data values
    times = problem.times()
    data = problem.values()

    # Dense time points for plotting simulated solutions
    dense_times = np.linspace(min(times), max(times), 1000)

    # Make full figure with large gridspec
    fig = plt.figure(figsize=figsize, constrained_layout=True)
    gs = fig.add_gridspec(8*problem.n_parameters(), 10)

    # Create all axes
    axes = []
    for i in range(problem.n_parameters()):
        # Ax for plotting likelihood profile
        ax = fig.add_subplot(gs[8*i:8*i+7, :5])
        axes.append(ax)

        # Ax for slider bar
        ax = fig.add_subplot(gs[8*i+7, :5])
        axes.append(ax)

    # Ax for plotting time series data
    axes.append(fig.add_subplot(gs[2:, 5:]))

    # Ax for choosing tolerance/step size
    axes.append(fig.add_subplot(gs[:2, 5:6]))

    if tolerances is None:
        tolerances = step_sizes
        step = True

    else:
        step = False

    # Precalculate likelihood profiles for all parameters and solver settings
    likelihood_profiles = {}
    for tolerance in tolerances:
        if step:
            problem._model.set_step_size(tolerance)
        else:
            problem._model.set_tolerance(tolerance)

        likelihood_profiles[tolerance] = []

        for i in range(problem.n_parameters()):
            true_value = true_params[i]
            prange = np.linspace(0.8*true_value, 1.2*true_value, 100)
            lls = []
            for p in prange:
                v = true_params.copy()
                v[i] = p
                lls.append(likelihood(v))

            likelihood_profiles[tolerance].append((prange, lls))

    # Plot time series and data
    ax = axes[-2]
    ax.scatter(times,
               data,
               label='Data',
               color='red',
               s=2.5,
               zorder=-10,
               alpha=0.85)

    # Plot the ODE solution
    if step:
        problem._model.set_step_size(tolerances[0])
    else:
        problem._model.set_tolerance(tolerances[0])
    dataline, = ax.plot(
        dense_times,
        problem._model.simulate(true_params, dense_times),
        lw=1.5,
        color='k',
        label='Solution')

    ax.set_xlabel('Time')
    ax.legend()

    # Plot the likelihoods
    sliders = []
    vlines = []
    lplots = []
    for i in range(problem.n_parameters()):
        ax = axes[2*i]
        ax.set_ylabel('Loglikelihood')

        # Plot likelihood
        prange, lls = likelihood_profiles[tolerances[0]][i]
        lplot, = ax.plot(prange, lls)
        lplots.append(lplot)

        # Plot vertical line at parameter value
        vline = ax.axvline(true_params[i], color='blue', zorder=-10)
        vlines.append(vline)
        ax.axvline(
            true_params[i],
            color='k',
            ls='--',
            zorder=-9,
            label='Ground truth')

        ax.legend()

        p_slider_ax = axes[2*i+1]
        p_slider = Slider(
            p_slider_ax,
            param_names[i],
            min(prange),
            max(prange),
            valinit=true_params[i])
        sliders.append(p_slider)

    # Update function for sliders
    def update(val):
        v = true_params.copy()
        for i in range(problem.n_parameters()):
            p = sliders[i].val
            if p == val:
                v[i] = p
            else:
                p = true_params[i]
                if p != sliders[i].val:
                    sliders[i].set_val(true_params[i])
            v[i] = p
            vlines[i].set_xdata(p)

        y = problem._model.simulate(v, dense_times)

        dataline.set_xdata(dense_times)
        dataline.set_ydata(y)

        for ax in axes:
            ax.relim()
            ax.autoscale_view()

        fig.canvas.draw_idle()

    for slider in sliders:
        slider.on_changed(update)

    # Choosing tolerance with buttons
    radios_ax = axes[-1]
    radios_ax.set_title('Step size' if step else 'Tolerance')
    radios = RadioButtons(radios_ax, tolerances, active=0)

    # Function for choosing tolerance
    def radios_on_clicked(label):
        tol = float(label)

        if step:
            problem._model.set_step_size(tol)
        else:
            problem._model.set_tolerance(tol)

        for i in range(problem.n_parameters()):
            prange, lls = likelihood_profiles[tol][i]
            lplots[i].set_xdata(prange)
            lplots[i].set_ydata(lls)
            vlines[i].set_xdata(true_params[i])

        y = problem._model.simulate(true_params, dense_times)
        dataline.set_xdata(dense_times)
        dataline.set_ydata(y)

        for ax in axes:
            ax.relim()
            ax.autoscale_view()

        fig.canvas.draw_idle()

    radios.on_clicked(radios_on_clicked)

    for ax in axes:
        ax.relim()
        ax.autoscale_view()

    plt.show()


def plot_grouped_parameter_posteriors(
        *chains,
        true_model_parameters=None,
        colors=['white', 'lightblue', 'grey', 'darkblue'],
        method_names=None,
        parameter_names=None,
        fname='posterior.pdf'):
    """Plot the posterior distributions from different methods.

    This figure creates one panel for each model parameter, and groups the
    posteriors for each method by replicate along the x axis.

    Parameters
    ----------
    *chains : np.ndarry
        MCMC chains containing the samples. For each method, the MCMC chains in
        one numpy array should be provided.
    true_model_parameters : list of float, optional (None)
        The ground truth values of the model parameters, to be drawn on the
        plot.
    colors : list of str, optional
        Colors to use for labelling the posterior from each method. Must have
        length equal to the number of chains.
    method_names : list of str, optional (None)
        List giving the name of each method.
    parameter_names : list of str, optional (None)
        List giving the name of each model parameter.
    fname : str, optional ('posterior.pdf')
        Filename to save the figure. If None, the figure is not saved but
        returned.
    """
    num_model_parameters = len(true_model_parameters)

    fig = plt.figure(figsize=(6, 4.5))

    for i in range(num_model_parameters):
        # Make one panel for this model parameter
        ax = fig.add_subplot(num_model_parameters, 1, i+1)
        legend_boxes = []

        for j in range(len(chains)):
            # Plot the posterior from this method
            all_samplers = np.array(chains[j])
            samples = all_samplers[:, :, i]
            num_runs = all_samplers.shape[0]
            positions = (np.arange(num_runs) * (len(chains) + 2)) + 1 + j

            # Get the middle chain to use as the location of the run label
            if j == len(chains) // 2:
                tick_positions = positions

            # Settings for boxplot
            medianprops = dict(linewidth=0)

            # Plot all runs from this method and model parameter
            boxes = ax.boxplot(samples.T,
                               positions=positions,
                               sym='',
                               whis=[2.5, 97.5],
                               medianprops=medianprops,
                               patch_artist=True)

            for patch in boxes['boxes']:
                patch.set_facecolor(colors[j])

            # Add a box of the appropriate color to the legend
            legend_box = mpatches.Patch(facecolor=colors[j],
                                        label=method_names[j],
                                        edgecolor='black',
                                        linewidth=1,
                                        linestyle='-')
            legend_boxes.append(legend_box)

        if i == 0:
            # Add a legend above the plot
            ax.legend(handles=legend_boxes,
                      loc='upper center',
                      bbox_to_anchor=(0.5, 1.2),
                      ncol=4)

        ax.axhline(true_model_parameters[i], ls='--', color='k')
        ax.set_xticks(tick_positions)
        ax.set_xticklabels([])
        ax.set_ylabel(parameter_names[i])

    ax.set_xlabel('Replicate')
    ax.set_xticklabels(np.arange(num_runs) + 1)
    fig.set_tight_layout(True)

    if fname is not None:
        plt.savefig(fname)

    return fig
