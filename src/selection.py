# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 16:36:56 2020

@author: Florian
"""
import numpy as np


def fitness_proportionate_selection(fitness, N):
    """
    "roulette wheel selection"
    a roulette wheel is build, where each individual occupies space according
    to its fitness.
    A random  number from the wheel is chosen.
    Now, for each random position, find that individual, that "occupies" the
    corresponding part of the wheel
    note: roulette_wheel contains the right side of occupied wheel part
    e.g.
    normalized fitness = [0.15095256, 0.52290646, 0.32614097]
    roulette_wheel = [0.15095256, 0.67385903, 1.   ]
    rand wheel poition = 0.0245
    first position that is bigger than this value = 0
    rand wheel poition = 0.3652
    first position that is bigger than this value = 1
    rand wheel poition = 0.7966
    first position that is bigger than this value = 2
    """
    fitness_n = fitness.copy()  # 100
    fitness_n = fitness_n/np.sum(fitness)  # 100
    roulette_wheel = np.cumsum(fitness_n)  # 100
    rand_wheel_pos = np.random.rand(N)  # 6
    indxs = np.empty((N,), dtype=np.int32)  # 6
    all_indxs = np.arange(len(fitness))  # 100
    for i in range(N):
        indxs[i] = all_indxs[rand_wheel_pos[i] < roulette_wheel][0]
    return indxs.reshape(-1, 2)


def uniform_selection(fitness, N):
    """Parents are selected by uniform random distribution"""
    indxs = np.empty((N,), dtype=np.int32)
    for i in range(N):
        indxs[i] = np.random.randint(0, high=len(fitness))
    return indxs.reshape(-1, 2)


def rank_based_selection(fitness, N):
    """ rank based selection. e.g.:
    fitness = [0.10411715, 0.36066647, 0.22495058]
    ranks = [0, 2, 1]"""
    all_ranks = np.arange(len(fitness))
    ranks = all_ranks[np.argsort(fitness)]
    return fitness_proportionate_selection(ranks, N)


def elitism(population, survivors, children,
            fitness_popu, fitness_surv, fitness_chrn):
    n_survivors = len(survivors)
    n_children = len(children)
    id_all_individuals = np.arange(n_survivors+n_children)
    population[:n_survivors, :] = survivors  # shallow copy!
    population[n_survivors:, :] = children  # shallow copy!
    fitness_popu[:n_survivors] = fitness_surv  # shallow copy!
    fitness_popu[n_survivors:] = fitness_chrn  # shallow copy!
    id_decreasing = np.argsort(-fitness_popu)  # highest fitness first
    id_best_individuals = id_all_individuals[id_decreasing[:n_survivors]]
    survivors[:] = population[id_best_individuals]
    fitness_surv[:] = fitness_popu[id_best_individuals]
    return None
