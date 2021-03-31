# -*- coding: utf-8 -*-
"""
Created on Sun Mar 28 12:11:42 2021

@author: Wilms
"""
# Load some libraries first
import numpy as np
import pygmo as pg

def get_target(problem_name, kwargs_problem = None):
    """
    Get optimal vector and function value given test function name..

    Parameters
    ----------
    problem_name : str
        Problem name as a string. Either from `pygmo` library or user defined.
    kwargs_problem : dictionary, optional
        Arguments passed to the test function. The default is None.

    Returns
    -------
    target : list
        List containing optimal vector and best function value as arrays.

    """
    
    if kwargs_problem is not None:
        try:
            target = [getattr(pg, problem_name)(**kwargs_problem).best_known()]
            f_best = pg.problem(getattr(pg, problem_name)(**kwargs_problem)).fitness(target[0])
        except AttributeError:
            target = [eval(problem_name)(**kwargs_problem).best_known()]
            if target[0].ndim == 1:
                f_best = pg.problem(eval(problem_name)(**kwargs_problem)).fitness(target[0])
            else:
                f_best = pg.problem(eval(problem_name)(**kwargs_problem)).fitness(target[0][0])
    else:
        try:
            target = [getattr(pg, problem_name)().best_known()]
            f_best = pg.problem(getattr(pg, problem_name)()).fitness(target[0])
        except AttributeError:
            target = [eval(problem_name)().best_known()]
            if target[0].ndim == 1:
                f_best = pg.problem(eval(problem_name)()).fitness(target[0])
            else:
                f_best = pg.problem(eval(problem_name)()).fitness(target[0][0])            
        
    target.append(f_best)
    
    return target

# Functions needed for aggregation in describe method
def q25(x):
    return x.quantile(0.25)

def q75(x):
    return x.quantile(0.75)

def iqr(x):
    return x.quantile(0.75) - x.quantile(0.25)

# User defined problems
class beale:
    
    def __init__(self):
        self.dim = 2
        
    def fitness(self, x):
        return [
            (1.5 - x[0] + x[0] * x[1]) ** 2 + 
            (2.25 -x[0] + x[0] * x[1] ** 2) ** 2 +
            (2.625 - x[0] + x[0] * x[1] ** 3) ** 2
        ]
    
    def get_bounds(self):
        return ([-4.5] * 2, [4.5] * 2)
    
    def get_name(self):
        return "Beale Function"
    
    def best_known(self):
        return np.array([3, 0.5])
    
    def gradient(self, x) :
        return pg.estimate_gradient(lambda x: self.fitness(x), x)
    
class goldstein_price:
    
    def __init__(self):
        self.dim = 2
        
    def fitness(self, x):
        return [
            ((1 + (x[0] + x[1] + 1) ** 2) * 
            (19 - 14 * x[0] + 3 * x[0] ** 2 - 14 * x[1] + 6 * np.prod(x) + 3 * x[1] ** 2)) *
            (30 + (2 * x[0] - 3 * x[1]) ** 2 * (18 - 32 * x[0] + 12 * x[0] ** 2 + 48 * x[1] - 36 * np.prod(x) + 27 * x[1] ** 2))
        ]
    
    def get_bounds(self):
        return ([-2] * 2, [2] * 2)
    
    def get_name(self):
        return "Goldstein-Price Function"
    
    def best_known(self):
        return np.array([0, -1])
    
    def gradient(self, x):
        return pg.estimate_gradient(lambda x: self.fitness(x), x)
    
class booth:
    
    def __init__(self):
        self.dim = 2
        
    def fitness(self, x):
        return [
            (x[0] + 2 * x[1] - 7) ** 2 + (2 * x[0] + x[1] - 5) ** 2
        ]
    
    def get_bounds(self):
        return ([-10] * 2, [10] * 2)
    
    def get_name(self):
        return "Booth Function"
    
    def best_known(self):
        return np.array([1, 3])
    
    def gradient(self, x):
        return pg.estimate_gradient(lambda x: self.fitness(x), x)

class bukin_n6:
    
    def __init__(self):
        self.dim = 2
        
    def fitness(self, x):
        return [
            100 * np.sqrt(np.abs(x[1] - 0.01 * x[0] ** 2)) + 0.01 * np.abs(x[0] + 10)
        ]
        
    def get_bounds(self):
        return ([-15, -3], [-5, 3])
    
    def get_name(self):
        return "Bukin Function N.6"
    
    def best_known(self):
        return np.array([-10, 1])
    
    def gradient(self, x):
        return pg.estimate_gradient(lambda x: self.fitness(x), x)

class matyas:
    
    def __init__(self):
        self.dim = 2
        
    def fitness(self, x):
        return [
            0.26 * (x[0] ** 2 + x[1] ** 2) - 0.48 * np.prod(x)
        ]
    
    def get_bounds(self):
        return ([-10] * 2, [10] * 2)
    
    def get_name(self):
        return "Matyas Function"
    
    def best_known(self):
        return np.array([0, 0])
    
    def gradient(self, x):
        return pg.estimate_gradient(lambda x: self.fitness(x), x)

class levi_n13:
    
    def __init__(self):
        self.dim = 2
        
    def fitness(self, x):
        return [
            (
                np.sin(3 * np.pi * x[0]) ** 2 + 
                (x[0] - 1) ** 2 * (1 + np.sin(3 * np.pi * x[1]) ** 2) + 
                (x[1] - 1) ** 2 * (1 + np.sin(2 * np.pi * x [1]) ** 2)
            )
        ]
    
    def get_bounds(self):
        return ([-10] * 2, [10] * 2)
    
    def get_name(self):
        return "Lévi Function N.13"
    
    def best_known(self):
        return np.array([1, 1])
    
    def gradient(self, x):
        return pg.estimate_gradient(lambda x: self.fitness(x), x)
    
class himmelblau:

    def __init__(self):
        self.dim = 2
        
    def fitness(self, x):
        return [
            (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2
        ]
    
    def get_bounds(self):
        return ([-5] * 2, [5] * 2)
    
    def get_name(self):
        return "Himmelblau's Function"
    
    def best_known(self):
        return np.array([
            [3, 2],
            [-2.805118, 3.131312],
            [-3.779310, -3.283186],
            [3.584428, -1.848126]
        ])
    
    def gradient(self, x):
        return pg.estimate_gradient(lambda x: self.fitness(x), x)
    
class camel:
    
    def __init__(self):
        self.dim = 2
        
    def fitness(self, x):
        return [
            2 * x[0] ** 2 - 1.05 * x[0] ** 4 + x[0] ** 6 / 6 + np.prod(x) + x[1] ** 2
        ]
    
    def get_bounds(self):
        return ([-5] * 2, [5] * 2)
    
    def get_name(self):
        return "Three Hump Camel Function"
    
    def best_known(self):
        return np.array([0, 0])
    
    def gradient(self, x):
        return pg.estimate_gradient(lambda x: self.fitness(x), x)
    
class easom:
    
    def __init__(self):
        self.dim = 2
        
    def fitness(self, x):
        return [
            (
                -np.cos(x[0]) * 
                np.cos(x[1]) * 
                np.exp(-((x[0] - np.pi) ** 2 + (x[1] - np.pi) ** 2))
            )
        ]
    
    def get_bounds(self):
        return ([-100] * 2, [100] * 2)
    
    def get_name(self):
        return "Easom Function"
    
    def best_known(self):
        return np.array([np.pi, np.pi])
    
    def gradient(self, x):
        return pg.estimate_gradient(lambda x: self.fitness(x), x)
    
class cross_in_tray:
    
    def __init__(self):
        self.dim = 2
        
    def fitness(self, x):
        return [
            -0.0001 * (np.abs(
                np.sin(x[0]) *
                np.sin(x[1]) * 
                np.exp(
                    np.abs(100 - np.sqrt(x[0] ** 2 + x[1] ** 2) / np.pi) +
                    1
                )
            ) ** 0.1)
        ]
    
    def get_bounds(self):
        return ([-10] * 2, [10] * 2)
    
    def get_name(self):
        return "Cross-in-Tray Function"
    
    def best_known(self):
        return np.array([
            [1.34941, -1.34941],
            [1.34941, 1.34941],
            [-1.34941, 1.34941],
            [-1.34941, -1.34941]
        ])
    
    def gradient(self, x):
        return pg.estimate_gradient(lambda x: self.fitness(x), x)
    
class eggholder:
    
    def __init__(self):
        self.dim = 2
        
    def fitness(self, x):
        return [
            -(x[1] + 47) *
            np.sin(
                np.sqrt(
                    np.abs(
                        x[0] / 2 + (x[1] + 47)
                    )
                )
            ) - 
            x[0] * np.sin(
                np.sqrt(
                    np.abs(
                        x[0] - (x[1] + 47)
                    )
                )
            )
        ]
    
    def get_bounds(self):
        return ([-512] * 2, [512] * 2)
    
    def get_name(self):
        return "Eggholder Function"
    
    def best_known(self):
        return np.array([512, 404.2319])
    
    def gradient(self, x):
        return pg.estimate_gradient(lambda x: self.fitness(x), x)
    
class h_table:
    
    def __init__(self):
        self.dim = 2
        
    def fitness(self, x):
        return [
            - np.abs(
                np.sin(x[0]) *
                np.cos(x[1]) *
                np.exp(
                    np.abs(
                        1 - np.sqrt(x[0] ** 2 + x[1] ** 2) / np.pi
                    )
                )
            )
        ]
    
    def get_bounds(self):
        return ([-10] * 2, [10] * 2)
    
    def get_name(self):
        return "Hölder Table Function"
    
    def best_known(self):
        return np.array([
            [8.05502, 9.66459],
            [-8.05502, 9.66459],
            [8.05502, -9.66459],
            [-8.05502, -9.66459]
        ])
    
    def gradient(self, x):
        return pg.estimate_gradient(lambda x: self.fitness(x), x)
    
class mccormick:
    
    def __init__(self):
        self.dim = 2
        
    def fitness(self, x):
        return [
            np.sin(np.sum(x)) +
            (x[0] - x[1]) ** 2 + 
            1.5 * x[0] + 2.5 * x[1] + 1
        ]
    
    def get_bounds(self):
        return ([-1.5, -3], [4] * 2)
    
    def get_name(self):
        return "McCormick Function"
    
    def best_known(self):
        return np.array([-0.54719, -1.54719])
    
    def gradient(self, x):
        return pg.estimate_gradient(lambda x: self.fitness(x), x)
    
class schaffer_n2:
    
    def __init__(self):
        self.dim = 2
        
    def fitness(self, x):
        return [
            0.5 + (np.sin(x[0] ** 2 - x[1] ** 2) ** 2 - 0.5) /
            (1 + 0.001 * (x[0] ** 2 + x[1] ** 2)) ** 2
        ]
    
    def get_bounds(self):
        return ([-100] * 2, [100] * 2)
    
    def get_name(self):
        return "Schaffer Function N.2"
    
    def best_known(self):
        return np.array([0, 0])
    
    def gradient(self, x):
        return pg.estimate_gradient(lambda x: self.fitness(x), x)
    
class schaffer_n4:

    def __init__(self):
        self.dim = 2
        
    def fitness(self, x):
        return [
            0.5 + 
            (np.cos(
                np.sin(
                    np.abs(
                        x[0] ** 2 - x[1] ** 2
                    )
                )
            ) ** 2 - 0.5) / 
            (1 + 0.001 * (x[0] ** 2 + x[1] ** 2)) ** 2
        ]
    
    def get_bounds(self):
        return ([-100] * 2, [100] * 2)
    
    def get_name(self):
        return "Schaffer Function N.4"
    
    def best_known(self):
        return np.array([
            [0, 1.25313],
            [0, -1.25313]
        ])
    
    def gradient(self, x):
        return pg.estimate_gradient(lambda x: self.fitness(x), x)
    
class styblinski_tang:
    
    def __init__(self, dim = 4):
        self.dim = dim
        
    def fitness(self, x):
        return [
            np.sum(
                np.power(x, 4) - 16 * np.power(x, 2) + 5 * x
            ) / 2
        ]
    
    def get_bounds(self):
        return ([-5] * self.dim, [5] * self.dim)
    
    def get_name(self):
        return "Styblinski-Tang Function"
    
    def best_known(self):
        return np.array([-2.903534] * self.dim)
    
    def gradient(self, x):
        return pg.estimate_gradient(lambda x: self.fitness(x), x)
