# -*- coding: utf-8 -*-
"""
Created on Sun Mar 28 12:10:12 2021

@author: Wilms
"""
import itertools
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pygmo as pg
import utils

matplotlib.rcParams['text.usetex'] = True

# %% Accuracy function
def acc(population, x0, f0, target, f_tol = 1e-06, x_tol = 1e-04):
    """Compute accuracy measures given population, starting values, target values and tolerance
    
    Compute varios accuracy measures for assesment of algorithm performance for both 
    function value and decision vector. Absolute deviation, absolute normed deviation,
    log10 of absolute normed deviation and whether the algorithm converged in f and x.
    
    Parameters
    ---------- 
    population: pygmo population
        A population class from the pygmo library.
    x0: array_like
        Starting values of decision vector of an algorithm.
    f0: float
        Function value of the starting decision vector x0.
    target: list
        List that contains an array of the optimal decision vector, as the first entry
        and the function value at the optimum as the second.
    f_tol: float, default = 1e-06
        Deviation from optimum that is considered as acceptable for convergence.
    x_tol: float, default = 1e-04
        Deviation from optimal decision vector that is considered as acceptable for convergence.
        
    Returns
    -------
    ser_acc: Series
        Returns a pandas series with entries for the accuracy measures describe above.
        
    Notes
    -----
    The accuracy measures returned are defined as in [1]_
    -- math:: 
    f_{acc} = f(\bar{x}) - f(x^*)
    x_{acc} = ||\bar{x} - x^*||
    f_{acc}^n = \frac{f(\bar{x}) - f(x^*)}{f(x^0) - f(x^*)}
    x_{acc}^n = \frac{||\bar{x} - x^*||}{||x^0 - x^*||}
    f_{acc}^l = -\log_10(f_{acc}^n)
    x_{acc}^l = -\log_10(x_{acc}^n)
        
    References
    ----------
    .. [1] Vahid Beiranvand, Warren Hare and Yves Lucet, 
    "Best Practice for Comparing Optimization Algorithms", 
    Optimization and Engineering, vol. 18, pp. 815-848, 2017.
            
    """
    
    f_best = population.champion_f
    x_best = population.champion_x
    
    # Target is a list supplied by the user
    f_prime = target[1]
    x_prime = target[0]
    
    # Calculate accuracy measures
    f_acc_abs = np.abs(f_best - f_prime)
    f_acc_n = f_acc_abs / (f0 - f_prime)
    
    # If accuracy is smaller than 5.25569e-141 set f_acc_l equal to infinity.
    if f_acc_n <= 5.25569e-141:
        f_acc_l = np.inf
    else:
        f_acc_l = -np.log10(f_acc_n)
    
    # Account for multiple global minima
    if x_prime.ndim == 1:
        x_acc_abs = np.linalg.norm(x_best - x_prime)
        x_acc_n = x_acc_abs / np.linalg.norm(x0 - x_prime)    
    else:
        x_acc_abs = np.linalg.norm(x_best - x_prime, axis = 1).min()
        x_acc_n = x_acc_abs / np.linalg.norm(x0 - x_prime, axis = 1).min()    
        
    if x_acc_n <= 5.25569e-141:
        x_acc_l = np.inf
    else:
        x_acc_l = -np.log10(x_acc_n)
    
    # Boolean for convergence. Use stopping criteria from pygmo library
    converged_f = (f_acc_abs < f_tol)
    converged_x = (x_acc_abs < x_tol)
    
    list_acc_index = ["acc_abs_f", "acc_abs_x", "acc_norm_f", "acc_norm_x", "acc_log_f", "acc_log_x", "converged_f", "converged_x"]
    return pd.Series([f_acc_abs, x_acc_abs, f_acc_n, x_acc_n, f_acc_l, x_acc_l, converged_f, converged_x], 
                     index = list_acc_index,
                     dtype = float
                    )

# %% Iterate on problem for single algorithm
def get_results_algo_problem(
    problem, algorithm_name, 
    target, 
    kwargs_algorithm = None,                          
    gen = 1000, 
    pop_size = 100, 
    iterations = 100,
    seed = 2093111, 
    verbosity = 1, 
    f_tol = 1e-06, 
    x_tol = 1e-04
):
    """Compute for a given problem and algorithm perofrmance measures for several runs.
    
    Due to the stochastic nature of most algorithms it is important to test several
    starting points. This function computes accuracy measures using the `acc` function.
    
    Paramters
    ---------
    problem: pygmo problem class
        A `pygmo` object of the class problem.
    algorithm_name: str
        A string referencing an algorithm from the `pygmo` library.
    target: list
        List that contains the optimal decision vector, as the first entry
        and the function value at the optimum as the second.
    kwargs_algorithm: dictionary, default = None
        Keywords that are passed to the `pygmo` algorithm class to define the solver.
    gen: int, default = 1000
        The number of generations a population is evolved before the process is stopped.
    pop_size: int, default = 100
        The number of individuals in a generation. Every individual has a decision vector 
        and a fitness value which is the function evaluated at the decision vector.
    iterations: int, default = 100
        The number of experiments that is run.
    seed: int, default = 2093111
        For reproducability a seed is set to generate populations. The number of seeds 
        used equals the number of iterations. The seed for iteration one is equal to the 
        input. The numer is then progressing with each iteration.
    verbosity: int, default = 1
        The number of generations for which a log entry is added. The value of 1 implies
        an entry every generation. While a value of x > 1 adds an entry every x generations.
    f_tol: float, default = 1e-06
        Deviation from optimum that is considered as acceptable for convergence.
    x_tol: float, default = 1e-04
        Deviation from optimal decision vector that is considered as acceptable for convergence.
        
    Returns
    -------
    df_acc: DataFrame
        A pandas dataframe with accuracy measures computed by the `acc` function. Each row 
        is one iteration. Columns for identification are given as well.
    df_logs: DataFrame
        A pandas dataframe containing log entries and accuracy measures. 
        Each row is one generation for a given iteration. 
        Columns for identification are given as well.
        
    Notes
    -----
    This function builds on the applications provided by the `pygmo` library [1]_.
    Custom problems and algorithms can be easily added to `pygmo` 
    and then used for benchmarking exercises.
    
    References
    ----------
    .. [1] Francesco Biscani and Dario Izzo, 
        "A parallel global multiobjective framework for optimization: pagmo",
        Journal of Open Source Software, vol. 5, p. 2339, 2020.
        
    """
    
    list_acc = list()
    list_logs = list()
    
    for iter in range(iterations):
        
        # Generate a population that with size equal pop_size
        population = pg.population(problem, size = pop_size, seed = seed + iter)
        
        # Multiple starting points. Use best x0 at the starting point.
        x0 = population.champion_x
        f0 = population.champion_f

        if algorithm_name == "mbh":
            if not problem.has_gradient():
                kwargs_algorithm["algo"] = pg.nlopt("neldermead")
            if (problem.get_nc() > 0):
                kwargs_algorithm["algo"] = pg.nlopt("slsqp")
            algorithm = pg.algorithm(getattr(pg, algorithm_name)(seed = seed + iter, **kwargs_algorithm))
        elif algorithm_name == "naive":
            if problem.has_gradient() and problem.get_nc() == 0:
                uda = pg.nlopt("lbfgs")
            elif problem.has_gradient() and problem.get_nc() > 0:
                uda = pg.nlopt("slsqp")
            elif not problem.has_gradient() and problem.get_nc() == 0:
                uda = pg.nlopt("neldermead")
            else:
                uda = pg.nlopt("cobyla")
            algorithm = pg.algorithm(uda)
        elif kwargs_algorithm is None:
            algorithm = pg.algorithm(getattr(pg, algorithm_name)(gen, seed = seed + iter))
        else:
            algorithm = pg.algorithm(getattr(pg, algorithm_name)(gen, seed = seed + iter, **kwargs_algorithm))
        algorithm.set_verbosity(verbosity)
        
        population = algorithm.evolve(population)
        
        if algorithm_name == "naive":
            log = algorithm.extract(pg.nlopt).get_log()
        else:
            log = algorithm.extract(getattr(pg, algorithm_name)).get_log()
        
        # Performance profiles need all observations
        if algorithm_name == "bee_colony":
            df_log = pd.DataFrame(log).iloc[:, [0, 1, 3]]
        elif algorithm_name == "mbh":
            df_log = pd.DataFrame(log).iloc[:, 0:2]
            df_log = pd.concat([pd.DataFrame(np.ones(df_log.shape[0])), df_log], axis = 1)
        elif algorithm_name == "naive":
            df_log = pd.DataFrame(log).iloc[:, 0:2]
            df_log[2] = df_log[0]
            df_log = df_log.iloc[:, [0, 2, 1]]
        elif len(log) == 0:
            return None
        else:
            df_log = pd.DataFrame(log).iloc[:, 0:3]
        df_log.columns = ["gen", "f_evals", "best"]
        
        # Add column for iteration number and algorithm
        df_log["iteration"] = iter + 1
        df_log["algorithm"] = algorithm_name
        
        # A column for relative loss log and negative log_10 loss
        # Since the log file, doesn't return the best x vector can only compute function loss
        df_log["acc_norm_f"] = np.abs((df_log["best"] - target[1]) / (f0 - target[1]))
        
        # log10 returns -inf if acc_norm_f < 
        ser_acc_norm_f = df_log["acc_norm_f"]
        ser_acc_log_f = pd.Series(np.repeat(np.inf, len(ser_acc_norm_f)))
        ser_acc_log_f.loc[ser_acc_norm_f > 5.25569e-141] = -np.log10(ser_acc_norm_f)
        df_log["acc_log_f"] = ser_acc_log_f
        df_log["converged_f"] = np.abs(df_log["best"] - target[1]) < f_tol
        
        # Append results to lists
        list_logs.append(df_log)
        
        ser_acc = acc(population, x0, f0, target, f_tol, x_tol)
        ser_acc["iteration"] = iter + 1
        ser_acc["algorithm"] = algorithm_name
        ser_acc["f_eval"] = population.problem.get_fevals()
        ser_acc["g_eval"] = population.problem.get_gevals()
        ser_acc["h_eval"] = population.problem.get_hevals()
        
        # pygmo doesn't
        list_acc.append(ser_acc)
        
    df_acc = pd.DataFrame(list_acc)
    df_acc = df_acc.astype({"converged_f": bool, "converged_x": bool})
    df_logs = pd.concat(list_logs)
        
    return df_acc, df_logs

# %% Iterate on problem single algorithm for multiple popsizes
def get_results_popsize(
    problem, 
    algorithm_name, 
    target, 
    kwargs_algorithm = None,
    gen = 1000, 
    list_pop_size = [20, 50, 100, 250], 
    iterations = 100, 
    seed = 2093111, 
    verbosity = 1, 
    f_tol = 1e-06, 
    x_tol = 1e-04
):
    """Compute for a given problem, algorithm and population size perofrmance measures for several runs.
    
    Since the population size is a tuning parameter of interest as well this function computes
    accuracy measures for differing population sizes. Due to the stochastic nature of most 
    algorithms it is important to try out several starting points
    
    Paramters
    ---------
    problem: pygmo problem class
        A `pygmo` object of the class problem.
    algorithm_name: str
        A string referencing an algorithm from the `pygmo` library.
    target: list
        List that contains the optimal decision vector, as the first entry
        and the function value at the optimum as the second.
    kwargs_algorithm: dictionary, default = None
        Keywords that are passed to the `pygmo` algorithm class to define the solver.
    gen: int, default = 1000
        The number of generations a population is evolved before the process is stopped.
    list_pop_size: list, default = [20, 50, 100, 250]
        List with the number of individuals in a generation. Every individual has a decision vector 
        and a fitness value which is the function evaluated at the decision vector.
    iterations: int, default = 100
        The number of experiments that is run.
    seed: int, default = 2093111
        For reproducability a seed is set to generate populations. The number of seeds 
        used equals the number of iterations. The seed for iteration one is equal to the 
        input. The numer is then progressing with each iteration.
    verbosity: int, default = 1
        The number of generations for which a log entry is added. The value of 1 implies
        an entry every generation. While a value of x > 1 adds an entry every x generations.
    f_tol: float, default = 1e-06
        Deviation from optimum that is considered as acceptable for convergence.
    x_tol: float, default = 1e-04
        Deviation from optimal decision vector that is considered as acceptable for convergence.
        
    Returns
    -------
    df_acc: DataFrame
        A pandas dataframe with accuracy measures computed by the `acc` function. Each row 
        is one iteration. Columns for identification are given as well.
    df_logs: DataFrame
        A pandas dataframe containing log entries and accuracy measures. 
        Each row is one generation for a given iteration. 
        Columns for identification are given as well.
        
    Notes
    -----
    This function builds on the applications provided by the `pygmo` library [1]_.
    Custom problems and algorithms can be easily added to `pygmo` 
    and then used for benchmarking exercises.
    
    References
    ----------
    .. [1] Francesco Biscani and Dario Izzo, 
        "A parallel global multiobjective framework for optimization: pagmo",
        Journal of Open Source Software, vol. 5, p. 2339, 2020.
        
    """
    
    list_acc_popsize = list()
    list_logs_popsize = list()
    
    for pop_size in list_pop_size:
        
        # Store in differing
        df_acc, df_logs = get_results_algo_problem(
            problem,
            algorithm_name,
            target,
            kwargs_algorithm,
            gen,
            pop_size,
            iterations,
            seed,
            verbosity,
            f_tol,
            x_tol
        )
        
        # Set popsize as variable
        df_acc["pop_size"] = pop_size
        df_logs["pop_size"] = pop_size
        
        list_acc_popsize.append(df_acc)
        list_logs_popsize.append(df_logs)
    
    df_acc_popsize = pd.concat(list_acc_popsize)
    df_logs_popsize = pd.concat(list_logs_popsize)
    
    return df_acc_popsize, df_logs_popsize

# %% Benchmark everything
def get_results_all(
    list_problem_names,
    list_algorithm_names,
    kwargs_problem = None,
    kwargs_algorithm = None,
    gen = 1000,
    list_pop_size = [20, 50, 100, 250],
    iterations = 100,
    seed = 2093111,
    verbosity = 1, 
    f_tol = 1e-06, 
    x_tol = 1e-04
):
    """Compute for a set of given problems, algorithms and population size 
    performance measures for several runs.
    
    This function calculates performance measures for differing algorithms
    on multiple problems with differing population sizes. Due to the stochastic
    nature of most global optimizers, multiple runs are executed for a given
    problem algorithm and population size
    
    Paramters
    ---------
    list_problem_names: list of str
        A list containing the names of the problems used for benchmarking from the 
        `pygmo` library
    list_algorithm_names: list of str
        A list of strings referencing the algorithms to be benchmarked from the `pygmo` library.
    kwargs_problem: dicitonary, default = None
        A dictionary with keys as the strings from `list_problem_names`. For every
        key the value has to be a dictionary of arguments passed to the `pygmo` problem
        class. Keywords can be dimension or other parameters
    kwargs_algorithm: dictionary, default = None
        A dictionary with keys as the strings from `list_algorithm_names`. For every
        key the value has to be a dictionary of arguments passed to the `pygmo` algorithm
        class.
    gen: int, default = 1000
        The number of generations a population is evolved before the process is aborted.
    list_pop_size: list, default = [20, 50, 100, 250]
        List with the number of individuals in a generation. Every individual has a decision vector 
        and a fitness value which is the function evaluated at the decision vector.
    iterations: int, default = 100
        The number of experiments that is run.
    seed: int, default = 2093111
        For reproducability a seed is set to generate populations. The number of seeds 
        used equals the number of iterations. The seed for iteration one is equal to the 
        input. The numer is then progressing with each iteration.
    verbosity: int, default = 1
        The number of generations for which a log entry is added. The value of 1 implies
        an entry every generation. While a value of x > 1 adds an entry every x generations.
    f_tol: float, default = 1e-06
        Deviation from optimum that is considered as acceptable for convergence.
    x_tol: float, default = 1e-04
        Deviation from optimal decision vector that is considered as acceptable for convergence.
        
    Returns
    -------
    df_acc: DataFrame
        A pandas dataframe with accuracy measures computed by the `acc` function. Each row 
        is one iteration. Columns for identification are given as well.
    df_logs: DataFrame
        A pandas dataframe containing log entries and accuracy measures. 
        Each row is one generation for a given iteration. 
        Columns for identification are given as well.
        
    Notes
    -----
    This function builds on the applications provided by the `pygmo` library [1]_.
    Custom problems and algorithms can be easily added to `pygmo` 
    and then used for benchmarking exercises.
    
    References
    ----------
    .. [1] Francesco Biscani and Dario Izzo, 
        "A parallel global multiobjective framework for optimization: pagmo",
        Journal of Open Source Software, vol. 5, p. 2339, 2020.
        
    """
    
    if kwargs_problem is None:
        kwargs_problem = dict()
        for problem_name in list_problem_names:
            kwargs_problem[problem_name] = {"dim": None}
        
    if kwargs_algorithm is None:
        kwargs_algorithm = dict()
        for algorithm_name in list_algorithm_names:
            kwargs_algorithm[algorithm_name] = None
            
    # Check that a kwarg is present for every problem/algorithm
    if len(list_problem_names) != len(kwargs_problem):
        raise KeyError("kwargs_problem has to contain an entry for every problem. If you don't intend \
                       to supply any values set it equal to None")
        
    if len(list_algorithm_names) != len(kwargs_algorithm):
        raise KeyError("kwargs_algorithm has to contain an entry for every algorithm. If you don't intend \
                       to supply any values set it equal to None")
    
    # Containers for output
    list_acc_all = list()
    list_logs_all = list()
    
    # Greate a grid of problem and algorithm names over which to loop
    for problem_name, algorithm_name in itertools.product(list_problem_names, list_algorithm_names):
         
        # Define the problem
        if kwargs_problem[problem_name] is not None:
            try:
                problem = pg.problem(
                    getattr(pg, problem_name)(**kwargs_problem[problem_name])
                    )
            except AttributeError:
                    problem = pg.problem(
                        getattr(utils, problem_name)(**kwargs_problem[problem_name])
                        )
        else:
            try:
                problem = pg.problem(getattr(pg, problem_name)())
            except AttributeError:
                problem = pg.problem(getattr(utils, problem_name)())        
        
        # Get target for given problem
        target = utils.get_target(problem_name, kwargs_problem[problem_name])
        
        # Apply the get_results_popsize
        df_acc_problem_algorithm, df_logs_problem_algorithm = \
        get_results_popsize(
            problem,
            algorithm_name,
            target,
            kwargs_algorithm[algorithm_name],
            gen,
            list_pop_size,
            iterations,
            seed,
            verbosity
        )
        
        # Add a column to every dataframe identifying the problem considered
        df_acc_problem_algorithm["problem"] = problem_name
        df_logs_problem_algorithm["problem"] = problem_name
        
        # Put into list containers
        list_acc_all.append(df_acc_problem_algorithm)
        list_logs_all.append(df_logs_problem_algorithm)
        
        print("Currently at algorithm {0} and problem {1}".format(algorithm_name, problem_name))
        
    # Put into dataframe format
    df_acc_all = pd.concat(list_acc_all)
    df_logs_all = pd.concat(list_logs_all)
    
    return df_acc_all, df_logs_all

# %% Convergence Plot
def convergence_plot(
    df_logs, 
    problem, 
    algorithm = "all",
    metric = "median",
    subplot_kwargs = {"alpha": 0.75, "xscale": "linear", "yscale": "linear"},
    fig_kwargs = {"figsize": (21, 10.5), "dpi": 150},
    labels = None,
    label_kwargs = {"fontsize": 19}
):
    """Calculate a convergence plot from a log file.
    
    The convergence plot compares the best function value achieved by different 
    algorithm against the number of function evaluations. The plot compares
        multipe algorithms aggregated over all iterations on the same problem.
    
    Parameters
    ----------
    df_logs: Dataframe
        A pandas dataframe usually output of get_results_all. 
        The dataframe should contain the columns gen (generation), f_evals (function evaluations), 
        best (best current fitness value), iteration, algorithm, pop_size, problem.
    problem: str
        Define for which problem in the log file a plot is drawn.
    algorithm: list of str, default = "all"
        Contains the name of the benchmarked algorithms as strings. 
        By default all algorithms contained in the algorithm column of df_logs are used.
    metric: {"mean", "median"}, default = "median"
        The statistical estimator applied to df_logs for different iterations.
    subplot_kwargs: dictionary, default = {"alpha": 0.75, "xscale": "linear", "yscale": "linear"}
        Keyword arguments that are passed to the add_subplot call.
    fig_kwargs: dictionary, default = {"figsize": (21, 10.5), "dpi": 150}
        Keyword arguments that are passed to pyplot figure call.
    labels: dictionary, default = None
        Dictionary that passes a label for a given algorithm name.
    labels_kwargs: dictionary, default = {"fontsize": 19}
        Sets text properties that control the appearance of the labels. 
        All labels are set simultaneously.
    
    Returns
    -------
    fig: matplotlib.fig.Figure
        A matplotlib figure. How many figures are returned depends on the number of different
        population sizes in df_logs.
        
    Notes
    -----
    This function builds upon the pygmo library. However, every dataframe that fits the 
    structure of df_logs can passed to the function.
    
    The convergence plot is ill suited to compare the performance of many optimizers
    on a large sample of problems. Further, if many differing starting points are computed
    for an algorithm on a certain problem aggregation is difficult. The mean is prone to outliers
    and can therefore warp the actual performance of the optimizer. As a substitute the median is
    chosen. However, this will lead to an increase in function value after all
    iterations that did converge have terminated.

    """
    
    df_logs = df_logs.copy()
    
    # Need to filter the log dataframe depending on inputs
    if algorithm != "all":
        df_logs = df_logs[df_logs["algorithms"].isin(algorithm)]
    
    df_logs = df_logs[df_logs["problem"] == problem]
    
    # Find the number of subplots. Problems in the rows and popsize in the columns.
    arr_algorithms = df_logs["algorithm"].unique()
    arr_pop_size = df_logs["pop_size"].unique()
    
    cardinality_pop_size = arr_pop_size.shape[0]
    
    if labels is None:
        labels = dict(zip(arr_algorithms, [x.replace("_", " ").upper() for x in arr_algorithms]))
                    
    fig, ax = plt.subplots(
        ncols = cardinality_pop_size,
        sharex = False,
        sharey = "row",
        subplot_kw = subplot_kwargs,
        **fig_kwargs
    )
    
    # Only single x-label. Add one big frame
    fig.add_subplot(111, frameon = False)
    plt.tick_params(labelcolor = "none", top = False, bottom = False, left = False, right = False)
    plt.grid(False)
    plt.xlabel("Function Evaulations", **label_kwargs)
    plt.ylabel("Function Value", **label_kwargs)
    
    for col, algo in itertools.product(
        range(cardinality_pop_size),
        arr_algorithms
    ):
        
        df_logs_sub = (df_logs[
            (df_logs["pop_size"] == arr_pop_size[col]) &
            (df_logs["algorithm"] == algo)
        ])
        
        # Calculate mean, sd, and 95CI for every level of f_eval
        df_descriptives = (df_logs_sub.
                          groupby("f_evals").
                          aggregate({"best": {metric}}))
        
        df_descriptives.columns = df_descriptives.columns.droplevel()
        
        ax[col].plot(
            df_descriptives.index, 
            df_descriptives[metric],
            label = labels[algo]
        )
        
        ax[col].axhline(y = 0, color = "black")
        
        ax[col].set_title("Popsize: {}".format(arr_pop_size[col]), **label_kwargs)
        ax[col].tick_params(labelsize = 16)
        
    handles, labels = ax[0].get_legend_handles_labels()
    plt.legend(handles, labels, loc = "upper right", **label_kwargs)

    return fig

# %% Performance Profile
def performance_profile(
    df_acc,
    range_tau = 50,
    problem = "all",
    algorithm = "all",
    conv_measure = "f_value",
    subplot_kwargs = {"alpha": 0.75},
    fig_kwargs = {"figsize": (21, 10.5), "dpi": 150},
    labels = None,
    labels_kwargs = {"fontsize": 19}
):
    """Plot performance profiles given an accuracy data frame.
    
    A performance profile shows for what percentage of problems (for a given tolerance) 
    the candidate solution of an algorithm is among the best solvers.
    
    Parameters
    ----------
    df_acc: Dataframe
        A pandas dataframe usually output of get_results_all. 
        The dataframe should contain the columns acc_abs_f (absolute accuracy of f),
        acc_abs_x (absolute accurcay of x), acc_norm_f (normalized accuracy of f), 
        acc_norm_x (normalized accuracy of x), acc_log_f (log10 of normalized accuracy of f), 
        acc_log_x (log 10 of normalized accuracy of x), converged_f, converged_x,
        f_evals (function evaluations), iteration, algorithm, pop_size, problem.
    range_tau, int, default = 50
        Gives the range for which a performance profile is plotted.
    problem: str, default = "all"
        Define for which problems in df_acc file a plot is drawn.
    algorithm: list of str, default = "all"
        Contains the name of the benchmarked algorithms as strings. 
        By default all algorithms contained in the algorithm column of df_acc are used.
    conv_measure: {"f_value", "x_value"}, default = "f_value"
        Which measure to consider for deciding on convergence of algortihms
    subplot_kwargs: dictionary, default = {"alpha": 0.75, "xscale": "linear", "yscale": "linear"}
        Keyword arguments that are passed to the add_subplot call.
    fig_kwargs: dictionary, default = {"figsize": (21, 10.5), "dpi": 150}
        Keyword arguments that are passed to pyplot figure call.
    labels: dictionary, default = None
        Dictionary that passes a label for a given algorithm name.
    labels_kwargs: dictionary, default = {"fontsize": 19}
        Sets text properties that control the appearance of the labels. 
        All labels are set simultaneously.
    
    Returns
    -------
    fig: matplotlib.fig.Figure
        A matplotlib figure. How many figures are returned depends on the number of different
        population sizes in df_acc.
        
    Notes
    -----
    This function builds upon the pygmo library. However, every dataframe that fits the 
    structure of df_acc as outlined above can be passed to the function. 
    
    As an example, a ratio value of 10 for example would show us what percentage of a given algorithm
    achieve convergence in at least ten times the amount of evaluations of the
    best optimizer. It is important to consider that the best solution found, 
    doesn't have to be the correct global minimum.
    
    """
    
    df_acc = df_acc.copy()
    
    # Need to filter the log dataframe depending on inputs of problem and algorithm
    if problem != "all":
        df_acc = df_acc[df_acc["problem"].isin(problem)]
    if algorithm != "all":
        df_acc = df_acc[df_acc["algorithm"].isin(algorithm)]
        
    arr_algorithms = df_acc["algorithm"].unique()
    arr_problems = df_acc["problem"].unique()
    arr_pop_size = df_acc["pop_size"].unique()
    
    cardinality_problems = df_acc["problem"].nunique()
    cardinality_pop_size = df_acc["pop_size"].nunique()
    cardinality_iteration = df_acc["iteration"].nunique()
        
    df_acc["perf_ratio"] = np.inf
    df_acc["sum_eval"] = df_acc[["f_eval", "g_eval", "h_eval"]].sum(axis = 1)
    
    if conv_measure == "x_value":
        df_acc["converged"] = df_acc["converged_x"]
    else:
        df_acc["converged"] = df_acc["converged_f"]
    
    # Need to calculate performance ratio for algo, problem and pop_size    
    for algo, problem, pop in itertools.product(
        arr_algorithms,
        arr_problems,
        arr_pop_size
    ):
        
        str_query = ("problem == '{0}' \
                     and pop_size == {1} \
                     and converged == True".
                     format(problem, pop))
            
        fl_min_measure = (df_acc.loc[df_acc.eval(str_query), "sum_eval"].
                          dropna().
                          min())
        
        if fl_min_measure is np.nan:
            pass
        else:
            str_query_pr_best = (
                str_query + 
                " and algorithm == '{0}' \
                and sum_eval == {1}".
                format(algo, fl_min_measure)
            )
            
            df_acc.loc[df_acc.eval(str_query_pr_best), "perf_ratio"] = 1
            
            str_query_pr_rest = (
                str_query + 
                " and algorithm == '{0}' \
                and converged == True".
                format(algo)
            )
            
            df_acc.loc[df_acc.eval(str_query_pr_rest), "perf_ratio"] = (
                df_acc.loc[df_acc.eval(str_query_pr_rest), "sum_eval"] / 
                fl_min_measure        
            )
        
    # Create a grid for tau and compute performance profile given algo and pop
    arr_tau = np.arange(1, range_tau + 1)
    list_perf_profile = list()
    
    for algo, pop in itertools.product(arr_algorithms, arr_pop_size):
        
        df_perf_profile_sub = pd.DataFrame(
            data = arr_tau,
            columns = ["tau"]
        )
        
        df_perf_profile_sub["algorithm"] = algo
        df_perf_profile_sub["pop_size"] = pop
        df_perf_profile_sub["perf_profile"] = np.nan
        
        str_query = "algorithm == '{0}' and pop_size == {1}".format(algo, pop)
        
        for tau in arr_tau:
            
            df_perf_profile_sub.loc[
                df_perf_profile_sub["tau"] == tau, "perf_profile"] = (
                np.sum(df_acc.loc[df_acc.eval(str_query), "perf_ratio"] < tau) /
                (cardinality_problems * cardinality_iteration)
            )
            
        list_perf_profile.append(df_perf_profile_sub)
        
    df_perf_profile = pd.concat(list_perf_profile)
        
    # Creat one figure for every pop_size
    if labels is None:
        labels = dict(zip(arr_algorithms, [x.replace("_", " ").upper() for x in arr_algorithms]))

                    
    fig, ax = plt.subplots(
        ncols = cardinality_pop_size,
        sharex = False,
        sharey = "row",
        subplot_kw = subplot_kwargs,
        **fig_kwargs
    )
    
    # Only single x-label. Add one big frame
    fig.add_subplot(111, frameon = False)
    plt.tick_params(labelcolor = "none", top = False, bottom = False, left = False, right = False)
    plt.grid(False)
    plt.xlabel("Ratio", **labels_kwargs)
    plt.ylabel("Performance Profile", **labels_kwargs)
    
    for algo, col in itertools.product(arr_algorithms, range(cardinality_pop_size)):
        
        str_query = "algorithm == '{0}' and pop_size == {1}".format(algo, arr_pop_size[col])
        df_perf_profile_sub = df_perf_profile.query(str_query)
        ax[col].step(
            x = "tau", 
            y = "perf_profile", 
            data = df_perf_profile_sub, 
            label = labels[algo]
        )
        
        ax[col].axhline(y = 0, color = "black")
        ax[col].set_title("Popsize: {}".format(arr_pop_size[col]), **labels_kwargs)
        ax[col].tick_params(labelsize = 16)
        
    handles, labels = ax[0].get_legend_handles_labels()
    plt.legend(handles, labels, loc = "upper right", **labels_kwargs)

    return fig

# %% Accuracy Profile
def accuracy_profile(
    df_acc,
    range_tau = 15,
    problem = "all",
    algorithm = "all",
    profile_type = "f_value",
    max_improvement = 15,
    subplot_kwargs = {"alpha": 0.75},
    fig_kwargs = {"figsize": (21, 10.5), "dpi": 150},
    labels = None,
    labels_kwargs = {"fontsize": 19}
):
    """Plot accuracy profiles given an accuracy data frame.
    
    An accuracy profile shows for what proportion of problems
    an algorithm achieved an accuracy of at least a given value.
    
    Parameters
    ----------
    df_acc: Dataframe
        A pandas dataframe usually output of get_results_all. 
        The dataframe should contain the columns acc_abs_f (absolute accuracy of f),
        acc_abs_x (absolute accurcay of x), acc_norm_f (normalized accuracy of f), 
        acc_norm_x (normalized accuracy of x), acc_log_f (log10 of normalized accuracy of f), 
        acc_log_x (log 10 of normalized accuracy of x), converged_f, converged_x,
        f_evals (function evaluations), iteration, algorithm, pop_size, problem.
    range_tau, int, default = 15
        Gives the range for which an accuracy profile is plotted. Can be interpreted
        as the digits of accuracy achieved
    problem: str, default = "all"
        Define for which problems in df_acc file a plot is drawn.
    algorithm: list of str, default = "all"
        Contains the name of the benchmarked algorithms as strings. 
        By default all algorithms contained in the algorithm column of df_acc are used.
    profile_type: {"f_value", "x_value"}, default = "f_value"
        Determines whether a data profile is drawn for accuracy of the function value or for
        the decision vector.
    subplot_kwargs: dictionary, default = {"alpha": 0.75, "xscale": "linear", "yscale": "linear"}
        Keyword arguments that are passed to the add_subplot call.
    fig_kwargs: dictionary, default = {"figsize": (21, 10.5), "dpi": 150}
        Keyword arguments that are passed to pyplot figure call.
    labels: dictionary, default = None
        Dictionary that passes a label for a given algorithm name.
    labels_kwargs: dictionary, default = {"fontsize": 19}
        Sets text properties that control the appearance of the labels. 
        All labels are set simultaneously.
    
    Returns
    -------
    fig: matplotlib.fig.Figure
        A matplotlib figure. How many figures are returned depends on the number of different
        population sizes in df_acc.
        
    Notes
    -----
    This function builds upon the pygmo library. However, every dataframe that fits the 
    structure of df_acc as outlined above can be passed to the function.
    
    This function is only applicable for problems with a known global optimum.
    
    """
    
    df_acc = df_acc.copy()
    
    # Need to filter the log dataframe depending on inputs of problem and algorithm
    if problem != "all":
        df_acc = df_acc[df_acc["problem"].isin(problem)]
    if algorithm != "all":
        df_acc = df_acc[df_acc["algorithm"].isin(algorithm)]
        
    arr_algorithms = df_acc["algorithm"].unique()
    arr_pop_size = df_acc["pop_size"].unique()
    
    cardinality_problems = df_acc["problem"].nunique()
    cardinality_pop_size = df_acc["pop_size"].nunique()
    cardinality_iteration = df_acc["iteration"].nunique()
    
    if profile_type == "x_value":
        df_acc["accuracy"] = df_acc["acc_log_x"]
    else:
        df_acc["accuracy"] = df_acc["acc_log_f"]
        
    df_acc["gamma"] = max_improvement
    df_acc.loc[df_acc["accuracy"] <= max_improvement, "gamma"] = (
        df_acc.loc[df_acc["accuracy"] <= max_improvement, "accuracy"]
    )

    # Create a grid for tau and compute accuracy profile given algo and pop
    arr_tau = np.arange(0, range_tau + 1)
    list_acc_profile = list()

    for algo, pop in itertools.product(arr_algorithms, arr_pop_size):
        
        df_acc_profile_sub = pd.DataFrame(
            data = arr_tau,
            columns = ["tau"]
        )
        
        df_acc_profile_sub["algorithm"] = algo
        df_acc_profile_sub["pop_size"] = pop
        df_acc_profile_sub["acc_profile"] = np.nan
        
        str_query = "algorithm == '{0}' and pop_size == {1}".format(algo, pop)
        
        for tau in arr_tau:
            
            df_acc_profile_sub.loc[
                df_acc_profile_sub["tau"] == tau, "acc_profile"] = (
                np.sum(df_acc.loc[df_acc.eval(str_query), "gamma"] >= tau) /
                (cardinality_problems * cardinality_iteration)
            )
            
        list_acc_profile.append(df_acc_profile_sub)
        
    df_acc_profile = pd.concat(list_acc_profile)
        
    # Creat one figure for every pop_size
    if labels is None:
        labels = dict(zip(arr_algorithms, [x.replace("_", " ").upper() for x in arr_algorithms]))
                    
    fig, ax = plt.subplots(
        ncols = cardinality_pop_size,
        sharex = False,
        sharey = "row",
        subplot_kw = subplot_kwargs,
        **fig_kwargs
    )
    
    # Only single x-label. Add one big frame
    fig.add_subplot(111, frameon = False)
    plt.tick_params(labelcolor = "none", top = False, bottom = False, left = False, right = False)
    plt.grid(False)
    plt.xlabel("Digits of Accuracy", **labels_kwargs)
    plt.ylabel("Accuracy Profile", **labels_kwargs)
    
    for algo, col in itertools.product(arr_algorithms, range(cardinality_pop_size)):
        
        str_query = "algorithm == '{0}' and pop_size == {1}".format(algo, arr_pop_size[col])
        df_acc_profile_sub = df_acc_profile.query(str_query)
        ax[col].step(
            x = "tau", 
            y = "acc_profile", 
            data = df_acc_profile_sub, 
            label = labels[algo]
        )
        
        ax[col].axhline(y = 0, color = "black")
        ax[col].set_title("Popsize: {}".format(arr_pop_size[col]), **labels_kwargs)
        ax[col].tick_params(labelsize = 16)
        
    handles, labels = ax[0].get_legend_handles_labels()
    plt.legend(handles, labels, loc = "upper right")

    return fig

# %% Data Profile
def data_profile(
    df_acc,
    problem = "all",
    algorithm = "all",
    profile_type = "f_value",
    perf_measure = 1e-06,
    problem_kwargs = None,
    subplot_kwargs = {"alpha": 0.75},
    fig_kwargs = {"figsize": (21, 10.5), "dpi": 150},
    labels = None,
    labels_kwargs = {"fontsize": 19}
):
    """Plot data profiles given an accuracy data frame.
    
    A data profile shows what percentage of problems (for a given tolerance) 
    can be solved given a budget of $ k $ function evaluations.
    
    Parameters
    ----------
    df_acc: Dataframe
        A pandas dataframe usually output of get_results_all. 
        The dataframe should contain the columns converged_f and/or converged_x, 
        f_evals (function evaluations), iteration, algorithm, pop_size, problem.
    problem: str, default = "all"
        Define for which problems in df_acc file a plot is drawn.
    algorithm: list of str, default = "all"
        Contains the name of the benchmarked algorithms as strings. 
        By default all algorithms contained in the algorithm column of df_acc are used.
    profile_type: {"f_value", "x_value"}, default = "f_value"
        Determines whether a data profile is drawn for accuracy of the function value or for
        the decision vector.
    perf_measure: float, default = 1e-06
        Measure that determines whether convergence has been achieved.
    problem_kwargs: dictionary, default = None
        Keyword arguments passed to the test function.
    subplot_kwargs: dictionary, default = {"alpha": 0.75, "xscale": "linear", "yscale": "linear"}
        Keyword arguments that are passed to the add_subplot call.
    fig_kwargs: dictionary, default = {"figsize": (21, 10.5), "dpi": 150}
        Keyword arguments that are passed to pyplot figure call.
    labels_kwargs: dictionary, default = None
        Dictionary that passes a label for a given algorithm name.
    
    Returns
    -------
    fig: matplotlib.fig.Figure
        A matplotlib figure. How many figures are returned depends on the number of different
        population sizes in df_acc.
        
    Notes
    -----
    This function builds upon the pygmo library. However, every dataframe that fits the 
    structure of df_acc as outlined above can be passed to the function.
    
    To make algorithms that use gradients/hessians and algorithms that don't comparable,
    the number of gradient and hessian evaluations are considered as well.
    
    """
    
    df_acc = df_acc.copy()
    
    # Need to filter the log dataframe depending on inputs of problem and algorithm
    if problem != "all":
        df_acc = df_acc[df_acc["problem"].isin(problem)]
    if algorithm != "all":
        df_acc = df_acc[df_acc["algorithm"].isin(algorithm)]
        
    arr_algorithms = df_acc["algorithm"].unique()
    arr_problems = df_acc["problem"].unique()
    arr_pop_size = df_acc["pop_size"].unique()
    
    cardinality_problems = df_acc["problem"].nunique()
    cardinality_pop_size = df_acc["pop_size"].nunique()
    cardinality_iterations = df_acc["iteration"].nunique()
    
    df_acc["sum_eval"] = df_acc[["f_eval", "g_eval", "h_eval"]].sum(axis = 1)
    df_acc["conv_criterion"] = np.nan
    
    for problem in arr_problems:
        
        if problem_kwargs is None or problem_kwargs[problem] is None:
            int_dim_problem = 2
        elif problem_kwargs[problem]["dim"] is not None:
            int_dim_problem = problem_kwargs[problem]["dim"]
        else:
            int_dim_problem = 2
            
        if profile_type == "f_value":
            str_query = "problem == '{}' and converged_f == True".format(problem)
        else:
            str_query = "problem == '{}' and converged_x == True".format(problem)
        df_acc.loc[df_acc.eval(str_query), "conv_criterion"] = (
            df_acc.loc[df_acc.eval(str_query), "sum_eval"] / 
            (int_dim_problem + 1)
        )
        
    df_acc["conv_criterion"] = np.ceil(df_acc["conv_criterion"])
    
    list_data_profile = list()
    # For every pop build a grid of max function evaluations for every algorithm.
    for pop in arr_pop_size:
        
        str_query_pop = "pop_size == {}".format(pop)
        
        # conv. criterion still has to be divided by pop size
        df_acc.loc[df_acc.eval(str_query_pop), "conv_criterion"] = (
            df_acc.loc[df_acc.eval(str_query_pop), "conv_criterion"] / pop
        )
        
        # Gen is equal to f_eval divided by pop
        int_max_eval = df_acc.loc[df_acc.eval(str_query_pop), "f_eval"].max() / pop
        arr_eval_range = np.arange(0, int_max_eval + pop, pop)
        
        # Repeat this for every algorithm
        df_eval_algo = pd.DataFrame(np.tile(arr_eval_range, len(arr_algorithms)),
                                  columns = ["eval"])
        df_eval_algo["algorithm"] = np.repeat(arr_algorithms, len(arr_eval_range))
        df_eval_algo["pop_size"] = pop
        df_eval_algo["data_profile"] = np.nan
        
        for algo, evals in itertools.product(arr_algorithms, arr_eval_range):
            
            str_query_algo = "pop_size == {0} and algorithm == '{1}'".format(pop, algo)
            fl_data_profile = (np.sum(
                df_acc.loc[df_acc.eval(str_query_algo), "conv_criterion"] < evals
            ) / (cardinality_problems * cardinality_iterations)
                              )
            df_eval_algo.loc[
                df_eval_algo.eval(str_query_algo + " and eval == {}".format(evals)), 
                "data_profile"] = fl_data_profile
            
        list_data_profile.append(df_eval_algo)
        
    df_data_profile = pd.concat(list_data_profile)
    
    # Creat one figure for every pop_size
    if labels is None:
        labels = dict(zip(arr_algorithms, [x.replace("_", " ").upper() for x in arr_algorithms]))

                    
    fig, ax = plt.subplots(
        ncols = cardinality_pop_size,
        sharex = False,
        sharey = "row",
        subplot_kw = subplot_kwargs,
        **fig_kwargs
    )
    
    # Only single x-label. Add one big frame
    fig.add_subplot(111, frameon = False)
    plt.tick_params(labelcolor = "none", top = False, bottom = False, left = False, right = False)
    plt.grid(False)
    plt.xlabel("Function Evaulations", **labels_kwargs)
    plt.ylabel("Data Profile", **labels_kwargs)
    
    for algo, col in itertools.product(arr_algorithms, range(cardinality_pop_size)):
        
        str_query = "algorithm == '{0}' and pop_size == {1}".format(algo, arr_pop_size[col])
        df_data_profile_sub = df_data_profile.query(str_query)
        ax[col].step(
            x = "eval", 
            y = "data_profile", 
            data = df_data_profile_sub, 
            label = labels[algo]
        )
        
        ax[col].axhline(y = 0, color = "black")
        ax[col].set_title("Popsize: {}".format(arr_pop_size[col]), **labels_kwargs)
        ax[col].tick_params(labelsize = 16)
        
    handles, labels = ax[0].get_legend_handles_labels()
    plt.legend(handles, labels, loc = "upper right")

    return fig

# %% Descriptive
def descriptive(df_acc):
    """Calculate descriptive measure given a dataframe containing accuracy and evaluation measures
    
    For every problem and popuplation size descriptives are computed. If applicable
    median, mean, quantiles and standard deviation. Further, descriptives are aggregated by
    population and problem.
    
    Parameters
    ----------
    df_acc: DataFrame
        A pandas dataframe usually output of get_results_all. 
        The dataframe should contain the columns converged_f and/or converged_x, 
        f_evals (function evaluations), iteration, algorithm, pop_size, problem.
    what: iterable, default = ["eval", "acc", "conv", "competition"]
        For which variables descriptive metrics should be calculated.
        
    Returns
    -------
    df_descriptive: DataFrame
        A pandas dataframe that contains 25%, 50%, 75% quantile and mean for 
        numeric accuracy measures. Both for accuracy of the decision vector and
        function value. Metrics are grouped by algorithm, population size and 
        problems. Further descriptives for the number of evaluations are given.
        For convergence in function value and in x-vector the mean is calculated
        which is equivalent to the convergence probability.
    df_comp: DataFrame
        A pandas dataframe containing the competitive classification proposed by
        [1]_. Furhter the probability of convergence in function and x-value is added
        to put competitiveness into perspective.
        
    Notes
    -----
    An algorithm is considered competitive by [1]_ if its solving time :math:`T_{algo} \leq 2 T_{min}`.
    The algorithm is further said to be very competitive if :math:`T_{algo} \leq T_{min}`.
    Solving time is measured as the sum of function, gradient and hessian evaluations.
    
    References
    ----------
    .. [1] Stephen C. Billups, Steven P. Dirkse and Michael C. Ferris,
        "A Comparison of Large Scale Mixed Complementarity Problem Solvers",
        Copmutational Optimization and Applications, vol. 7, pp 3-25, 1997.
        
    """

    # Transform column labels and string columns first
    df_acc = df_acc.copy()
    df_acc["eval"] = df_acc[["f_eval", "g_eval", "h_eval"]].sum(axis = 1)
    df_acc["problem"] = df_acc["problem"].str.title()
    df_acc["algorithm"] = df_acc["algorithm"].str.upper()
    dict_mapper = dict(
        zip(
            df_acc.columns,
            [
                "Abs. Accuracy f-value", "Abs. Accuracy x-value", 
                "Norm. Accuracy f-value", "Norm. Accuracy x-value",
                "Log of Norm. Accuracy f-value", "Log of Norm. Accuracy x-value",
                "Convergence in f-value in %", "Convergence in x-value in %",
                "Iteration",
                "Algorithm",
                "Function Evaluations",
                "Gradient Evaluations",
                "Hessian Evaluations",
                "Population Size",
                "Problem",
                "Evaluations"
            ]
        )
    )
    df_acc.rename(dict_mapper, axis = 1, inplace = True)
    
    df_descriptives = (df_acc.
                       groupby(["Algorithm", "Population Size", "Problem"]).
                       aggregate(
                           {
                               "Abs. Accuracy f-value": [utils.q25, "median", "mean", utils.q75],
                               "Norm. Accuracy f-value": [utils.q25, "median", "mean", utils.q75],
                               "Log of Norm. Accuracy f-value": [utils.q25, "median", "mean", utils.q75],
                               "Abs. Accuracy x-value": [utils.q25, "median", "mean", utils.q75],
                               "Norm. Accuracy x-value": [utils.q25, "median", "mean", utils.q75],
                               "Log of Norm. Accuracy x-value": [utils.q25, "median", "mean", utils.q75],
                               "Evaluations": [utils.q25, "median", "mean", utils.q75],
                               "Convergence in f-value in %": ["mean"],
                               "Convergence in x-value in %": ["mean"]
                           }                           
                       ).transpose()
                      )
        
    df_descriptives.index.set_levels(["Mean", "Median", "25%", "75%"], level = 1, inplace = True)
    df_descriptives.loc[["Convergence in f-value in %", "Convergence in x-value in %"]] = (
        np.round(
            df_descriptives.loc[["Convergence in f-value in %", "Convergence in x-value in %"]] * 100, 2
        )
    )
    
    arr_algorithms = df_acc["Algorithm"].unique()
    arr_problems = df_acc["Problem"].unique()
    arr_pop_size = df_acc["Population Size"].unique()
    
    # Calculate total descriptives
    for algo in arr_algorithms:
        df_descriptives[algo, "All", "All"] = df_descriptives[algo].mean(axis = 1)
        
    for algo, pop in itertools.product(arr_algorithms, arr_pop_size):
        df_descriptives[algo, pop, "All"] = df_descriptives[algo, pop].mean(axis = 1)
        
    for algo, problem in itertools.product(arr_algorithms, arr_problems):
        df_descriptives[algo, "All", problem] = (
            df_descriptives.loc(axis = 1)[algo, :, problem].mean(axis = 1)
        )
        
    # Calculate competitiveness of algorithm overall
        # Column evaluations for converged all
    str_query = "`Convergence in f-value in %` == True or `Convergence in x-value in %` == True"
    df_compet = (
        df_acc.
        query(str_query).
        groupby("Algorithm").
        aggregate({"Evaluations": "mean"})
    )
    fl_eval_min = df_compet["Evaluations"].min()
    df_compet["Competitiveness"] = "Not competitive"
    df_compet.loc[df_compet["Evaluations"] <= (2 * fl_eval_min), "Competitiveness"] = "Competitive"
    df_compet.loc[df_compet["Evaluations"] <= (4 / 3 * fl_eval_min), "Competitiveness"] = "Very competitive"
    
    # Get Probability of convergence as extra column
    df_prob_conv = (
        df_descriptives.
        loc[
            (["Convergence in f-value in %", "Convergence in x-value in %"]), 
            (slice(None), "All", "All")
        ].
        transpose()
    )
    df_prob_conv.index = df_prob_conv.index.droplevel(["Population Size", "Problem"])
    df_prob_conv.columns = df_prob_conv.columns.droplevel(level = 1)
    
    # Merge dataframe
    df_compet = df_compet.merge(df_prob_conv, on = "Algorithm")
    
    return df_descriptives, df_compet

# %% Define Benchmark class
class benchmark:
    """Run and analyse benchmarking experiments.
    
    This class allows comparison of several global optimizers on different problems and
    for different population sizes and parameters settings. Experiments can be run and 
    analysed using kpis and graphics.
    
    Parameters
    ----------
    problem_names: list of str
        A list containing the names of the problems used for benchmarking from the 
        `pygmo` library
    algorithm_names: list of str
        A list of strings referencing the algorithms to be benchmarked from the `pygmo` library.
    kwargs_problem: dicitonary, default = None
        A dictionary with keys as the strings from `list_problem_names`. For every
        key the value has to be a dictionary of arguments passed to the `pygmo` problem
        class. Keywords can be dimension or other parameters
    kwargs_algorithm: dictionary, default = None
        A dictionary with keys as the strings from `list_algorithm_names`. For every
        key the value has to be a dictionary of arguments passed to the `pygmo` algorithm
        class.
    gen: int, default = 1000
        The number of generations a population is evolved before the process is aborted.
    pop_size: list, default = [20, 50, 100, 250]
        List with the number of individuals in a generation. Every individual has a decision vector 
        and a fitness value which is the function evaluated at the decision vector.
    iterations: int, default = 100
        The number of experiments that is run.
    seed: int, default = 2093111
        For reproducability a seed is set to generate populations. The number of seeds 
        used equals the number of iterations. The seed for iteration one is equal to the 
        input. The numer is then progressing with each iteration.
    verbosity: int, default = 1
        The number of generations for which a log entry is added. The value of 1 implies
        an entry every generation. While a value of x > 1 adds an entry every x generations.
    f_tol: float, default = 1e-06
        Deviation from optimum that is considered as acceptable for convergence.
    x_tol: float, default = 1e-04
        Deviation from optimal decision vector that is considered as acceptable for convergence.
    
    Notes
    -----
    
    References
    ----------
    
    Examples
    --------
    
    """
    
    def __init__(
        self,
        problem_names,
        algorithm_names,
        kwargs_problem = None,
        kwargs_algorithm = None,
        gen = 1000,
        pop_size = [20, 50, 100, 250],
        iterations = 100,
        seed = 2093,
        verbosity = 1,
        f_tol = 1e-06,
        x_tol = 1e-04
    ):
        """Initialize variables to run the benchmarking exercise
        
        """
        
        (self.problem_names,
         self.algorithm_names,
         self.kwargs_problem,
         self.kwargs_algorithm,
         self.gen,
         self.pop_size,
         self.iterations,
         self.seed,
         self.verbosity,
         self.f_tol,
         self.x_tol,
         self.accuracy,
         self.logs,
         self.descriptive,
         self.competition
        ) = (
            problem_names,
            algorithm_names,
            kwargs_problem,
            kwargs_algorithm,
            gen,
            pop_size,
            iterations,
            seed,
            verbosity,
            f_tol,
            x_tol,
            "Run experiment first!",
            "Run experiment first!",
            "Run experiment first!",
            "Run experiment first!"
        )
        
    def run_experiment(self):
        """Compute for a set of given problems, algorithms and population size 
        performance measures for several runs.

        This function calculates performance measures for differing algorithms
        on multiple problems with differing population sizes. Due to the stochastic
        nature of most global optimizers, multiple runs are executed for a given
        problem algorithm and population size. Parameters are passed directly 
        from the class to the method. By changing an instance you can change
        the experiment.

        Notes
        -----
        This function builds on the applications provided by the `pygmo` library [1]_.
        Custom problems and algorithms can be easily added to `pygmo` 
        and then used for benchmarking exercises.

        References
        ----------
        .. [1] Francesco Biscani and Dario Izzo, 
            "A parallel global multiobjective framework for optimization: pagmo",
            Journal of Open Source Software, vol. 5, p. 2339, 2020.

        """
        
        self.accuracy, self.logs = get_results_all(
            list_problem_names = self.problem_names,
            list_algorithm_names = self.algorithm_names,
            kwargs_problem = self.kwargs_problem,
            kwargs_algorithm = self.kwargs_algorithm,
            gen = self.gen,
            list_pop_size = self.pop_size,
            iterations = self.iterations,
            seed = self.seed,
            verbosity = self.verbosity,
            f_tol = self.f_tol,
            x_tol = self.x_tol
        )
        
    def get_descriptives(self):
        """Calculate descriptive measure given a dataframe containing accuracy and evaluation measures

        For every problem and popuplation size descriptives are computed. If applicable
        median, mean, quantiles and standard deviation. Further, descriptives are aggregated by
        population and problem. Parameters are passed directly from the class to the method. 
        By changing an instance you can change the experiment. To use this method execute the
        method `run_experiment` first.

        Notes
        -----
        An algorithm is considered competitive by [1]_ if its solving time :math:`T_{algo} \leq 2 T_{min}`.
        The algorithm is further said to be very competitive if :math:`T_{algo} \leq T_{min}`.
        Solving time is measured as the sum of function, gradient and hessian evaluations.

        References
        ----------
        .. [1] Stephen C. Billups, Steven P. Dirkse and Michael C. Ferris,
            "A Comparison of Large Scale Mixed Complementarity Problem Solvers",
            Copmutational Optimization and Applications, vol. 7, pp 3-25, 1997.

        """
        
        self.descriptive, self.competition = descriptive(self.accuracy)
        
    def convergence_plot(self, 
                         problem, 
                         algorithm = "all",
                         metric = "median",
                         subplot_kwargs = {"alpha": 0.75, "xscale": "linear", "yscale": "linear"},
                         fig_kwargs = {"figsize": (21, 10.5), "dpi": 150},
                         labels = None,
                         label_kwargs = {"fontsize": 19}
                        ):
        """Calculate a convergence plot for the log instance.

        The convergence plot compares the best function value achieved by different 
        algorithms against the number of function evaluations. The convergence is
        aggregated over all iterations on the same problem.
        
        Parameters
        ----------
        problem: str
            Define for which problem in the log file a plot is drawn.
        algorithm: list of str, default = "all"
            Contains the name of the benchmarked algorithms as strings. 
            By default all algorithms contained in the algorithm column of the logs instance are used.
        subplot_kwargs: dictionary, default = {"alpha": 0.75, "xscale": "linear", "yscale": "linear"}
            Keyword arguments that are passed to the add_subplot call.
        fig_kwargs: dictionary, default = {"figsize": (21, 10.5), "dpi": 150}
            Keyword arguments that are passed to pyplot figure call.
        labels: dictionary, default = None
            Dictionary that passes a label for a given algorithm name.
        labels_kwargs: dictionary, default = {"fontsize": 19}
            Sets text properties that control the appearance of the labels. 
            All labels are set simultaneously.

        Returns
        -------
        fig_conv_plot: matplotlib.fig.Figure
            A matplotlib figure. How many figures are returned depends on the number of different
            population sizes in the logs instance.
            
        Notes
        -----
        The convergence plot is ill suited to compare the performance of many optimizers
        on a large sample of problems. Further, if many differing starting points are computed
        for an algorithm on a certain problem aggregation is difficult. The mean is prone to outliers
        and can therefore warp the actual performance of the optimizer. As a substitute the median is
        chosen. However, this will lead to an increase in function value after all
        iterations that did converge have terminated.

        """
        fig_conv_plot = convergence_plot(
            self.logs,
            problem,
            algorithm,
            metric,
            subplot_kwargs,
            fig_kwargs, 
            labels,
            label_kwargs            
        )
        
        return fig_conv_plot
    
    def performance_profile(self,
                            range_tau = 50,    
                            problem = "all",
                            algorithm = "all",
                            conv_measure = "f_value",
                            subplot_kwargs = {"alpha": 0.75},
                            fig_kwargs = {"figsize": (21, 10.5), "dpi": 150},
                            labels = None,
                            labels_kwargs = {"fontsize": 19}
                           ):
        """Plot performance profiles for the accuracy instance.

        A performance profile shows for what percentage of problems (for a given tolerance) 
        the candidate solution of an algorithm is among the best solvers.

        Parameters
        ----------
        range_tau, int, default = 50
            Gives the range for which a performance profile is plotted.
        problem: str, default = "all"
            Define for which problems in the accuracy instance a plot is drawn.
        algorithm: list of str, default = "all"
            Contains the name of the benchmarked algorithms as strings. 
            By default all algorithms contained in the algorithm column of the accuracy instance are used.
        conv_measure: {"f_value", "x_value"}, default = "f_value"
            Which measure to consider for deciding on convergence of algortihms
        subplot_kwargs: dictionary, default = {"alpha": 0.75, "xscale": "linear", "yscale": "linear"}
            Keyword arguments that are passed to the add_subplot call.
        fig_kwargs: dictionary, default = {"figsize": (21, 10.5), "dpi": 150}
            Keyword arguments that are passed to pyplot figure call.
        labels: dictionary, default = None
            Dictionary that passes a label for a given algorithm name.
        labels_kwargs: dictionary, default = {"fontsize": 19}
            Sets text properties that control the appearance of the labels. 
            All labels are set simultaneously.

        Returns
        -------
        fig_perf_profile: matplotlib.fig.Figure
            A matplotlib figure. How many figures are returned depends on the number of different
            population sizes in df_acc.

        Notes
        -----
        As an example, a ratio value of 10 for example would show us what percentage of a given algorithm
        achieve convergence in at least ten times the amount of evaluations of the
        best optimizer. It is important to consider that the best solution found, 
        doesn't have to be the correct global minimum.

        """
        
        fig_perf_profile = performance_profile(
            self.accuracy,
            range_tau, 
            problem,
            algorithm,
            conv_measure,
            subplot_kwargs,
            fig_kwargs,
            labels,
            labels_kwargs
        )
        
        return fig_perf_profile
    
    def accuracy_profile(self,
                         range_tau = 15,
                         problem = "all",
                         algorithm = "all",
                         profile_type = "f_value",
                         max_improvement = 15,
                         subplot_kwargs = {"alpha": 0.75},
                         fig_kwargs = {"figsize": (21, 10.5), "dpi": 150},
                         labels = None,
                         labels_kwargs = {"fontsize": 19}
                        ):
        """Plot accuracy profiles for the accuracy instance.

        An accuracy profile shows for what proportion of problems
        an algorithm achieved an accuracy of at least a given value.

        Parameters
        ----------
        range_tau, int, default = 50
            Gives the range for which an accuracy profile is plotted. Can be interpreted
        as the digits of accuracy achieved
        problem: str, default = "all"
            Define for which problems in the accuracy file a plot is drawn.
        algorithm: list of str, default = "all"
            Contains the name of the benchmarked algorithms as strings. 
            By default all algorithms contained in the algorithm column of the accuracy instance are used.
        profile_type: {"f_value", "x_value"}, default = "f_value"
            Determines whether a data profile is drawn for accuracy of the function value or for
            the decision vector.
        subplot_kwargs: dictionary, default = {"alpha": 0.75, "xscale": "linear", "yscale": "linear"}
            Keyword arguments that are passed to the add_subplot call.
        fig_kwargs: dictionary, default = {"figsize": (21, 10.5), "dpi": 150}
            Keyword arguments that are passed to pyplot figure call.
        labels: dictionary, default = None
            Dictionary that passes a label for a given algorithm name.
        labels_kwargs: dictionary, default = {"fontsize": 19}
            Sets text properties that control the appearance of the labels. 
            All labels are set simultaneously.

        Returns
        -------
        fig_acc_profile: matplotlib.fig.Figure
            A matplotlib figure. How many figures are returned depends on the number of different
            population sizes in df_acc.

        Notes
        -----
        This function builds upon the pygmo library. However, every dataframe that fits the 
        structure of df_acc as outlined above can be passed to the function.

        This function is only applicable for problems with a known global optimum.

        """
        
        fig_acc_profile = accuracy_profile(
            self.accuracy,
            range_tau, 
            problem,
            algorithm,
            profile_type,
            max_improvement,
            subplot_kwargs,
            fig_kwargs,
            labels,
            labels_kwargs
        )
        
        return fig_acc_profile
    
    def data_profile(self,
                     problem = "all",
                     algorithm = "all",
                     profile_type = "f_value",
                     perf_measure = 1e-06,
                     subplot_kwargs = {"alpha": 0.75},
                     fig_kwargs = {"figsize": (21, 10.5), "dpi": 150},
                     labels = None,
                     labels_kwargs = {"fontsize": 19}
                    ):
        """Plot data profiles given an accuracy instance.

        A data profile shows what percentage of problems (for a given tolerance) 
        can be solved given a budget of $ k $ function evaluations.

        Parameters
        ----------
        problem: str, default = "all"
            Define for which problems in df_acc file a plot is drawn.
        algorithm: list of str, default = "all"
            Contains the name of the benchmarked algorithms as strings. 
            By default all algorithms contained in the algorithm column of the accuracy instance are used.
        profile_type: {"f_value", "x_value"}, default = "f_value"
            Determines whether a data profile is drawn for accuracy of the function value or for
            the decision vector.
        perf_measure: float, default = 1e-06
            Measure that determines whether convergence has been achieved.
        subplot_kwargs: dictionary, default = {"alpha": 0.75, "xscale": "linear", "yscale": "linear"}
            Keyword arguments that are passed to the add_subplot call.
        fig_kwargs: dictionary, default = {"figsize": (21, 10.5), "dpi": 150}
            Keyword arguments that are passed to pyplot figure call.
        labels_kwargs: dictionary, default = None
            Dictionary that passes a label for a given algorithm name.

        Returns
        -------
        fig_data_profile: matplotlib.fig.Figure
            A matplotlib figure. How many figures are returned depends on the number of different
            population sizes in df_acc.

        Notes
        -----
        To make algorithms that use gradients/hessians and algorithms that don't comparable,
        the number of gradient and hessian evaluations are considered as well.

        """
        
        fig_data_profile = data_profile(self.accuracy,
                                        problem,
                                        algorithm,
                                        profile_type,
                                        perf_measure,
                                        self.kwargs_problem,
                                        subplot_kwargs,
                                        fig_kwargs,
                                        labels,
                                        labels_kwargs
                                       )
        
        return fig_data_profile