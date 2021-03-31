# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 13:48:11 2021

@author: Wilms
"""
import benchmark as bmk
import pygmo as pg

list_problem_names = (
    [
        "rosenbrock", "rastrigin", "schwefel", "ackley", "griewank", 
        "beale", "goldstein_price", "booth", "bukin_n6", "matyas",
        "levi_n13", "himmelblau", "camel", "cross_in_tray",
        "eggholder", "h_table", "mccormick", "schaffer_n2", "schaffer_n4",
        "styblinski_tang"
    ]
)

list_algorithm_names = [
    "bee_colony", "de", "sea", "sade", "cmaes", "pso", "pso_gen", "mbh", "naive"
]

# UDP have mostly two dimensions
kwargs_problem = {
    "rosenbrock": {"dim": 5}, "rastrigin": {"dim": 10}, "schwefel": {"dim": 2}, 
    "ackley": {"dim": 4}, "griewank": {"dim": 6}, "beale": None, "goldstein_price": None,
    "booth": None, "bukin_n6": None, "matyas": None, "levi_n13": None, "himmelblau": None,
    "camel": None, "cross_in_tray": None, "eggholder": None, 
    "h_table": None, "mccormick": None, "schaffer_n2": None, "schaffer_n4": None,
    "styblinski_tang": {"dim": 8}
}

kwargs_algorithm = {
    "bee_colony": None, "de": None, "sea": None, "sade": None, "cmaes": None, 
    "pso": None, "pso_gen": None, "mbh": {"algo": pg.nlopt("lbfgs")},
    "naive": None
}

dict_labels = {
    "bee_colony": "Bee Colony", "de": "DE", "sea": "SEA", "sade": "SADE", 
    "cmaes": "CMAES", "pso": "PSO", "pso_gen": "gen. PSO", "mbh": "MBH",
    "naive": "Naive"
}

bmk_project = bmk.benchmark(
    list_problem_names,
    list_algorithm_names,
    kwargs_problem,
    kwargs_algorithm
)

bmk_project.iterations = 20
bmk_project.gen = 500
bmk_project.run_experiment()

# %% Results
bmk_project.accuracy.to_csv("../Data/Accuracy_Final.csv")
bmk_project.logs.to_csv("../Data/Logs_Final.csv")
bmk_project.get_descriptive()
bmk_project.descriptive.to_csv("../Data/Descriptives_Final.csv")
bmk_project.competition.to_csv("../Data/Competitiveness_Final.csv")

fig_perf_profile = bmk_project.performance_profile(range_tau = 25)
fig_acc_profile = bmk_project.accuracy_profile(range_tau = 15)
fig_data_profile = bmk_project.data_profile()

# Export as png
fig_perf_profile.savefig(
    "../Plots/Performance_Profiles/Performance_Profile_Final.png", 
    dpi = 200
    )

fig_acc_profile.savefig(
    "../Plots/Accuracy_Profiles/Accuracy_Profile_Final.png", 
    dpi = 200
    )

fig_data_profile.savefig(
    "../Plots/Data_Profiles/Data_Profile_Final.png", 
    dpi = 200
    )

# Recompute accuracy and data profiles for x.
fig_acc_profile_x = bmk_project.accuracy_profile(
    range_tau = 15, 
    profile_type = "x_value"
    )
fig_data_profile_x = bmk_project.data_profile(profile_type = "x_value")

# Export as png
fig_acc_profile_x.savefig(
    "../Plots/Accuracy_Profiles/Accuracy_Profile_x_Final.png", 
    dpi = 200
    )

fig_data_profile_x.savefig(
    "../Plots/Data_Profiles/Data_Profile_x_Final.png", 
    dpi = 200
    )