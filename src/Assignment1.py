# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 17:03:49 2020

@author: Florian
"""
import numpy as np
from itertools import permutations
import matplotlib.pyplot as plt
import pickle
import time
from pprint import pprint
from selection import elitism, rank_based_selection
#from selection import fitness_proportionate_selection, uniform_selection
from recombinations_and_mutations import mutate, \
                    pmx_pair, cycle_crossover_pair, order_crossover_pair


def factorial(n):
    fact = 1
    for num in range(2, n + 1):
        fact *= num
    return fact


def travel_distance(distances, order):
    d = 0.0
    for i in range(len(order)):
        c1 = order[i-1]
        c2 = order[i]
        d += distances[c1, c2]
    return d


def exhaustive_search(distances):
    N = len(distances)
    opt_dist = np.inf
    opt_order = None
    for i, order in enumerate(permutations(np.arange(N))):
        d = travel_distance(distances, order)
        if d < opt_dist:
            opt_dist = d
            opt_order = order
    return opt_order, opt_dist


def hillclimb_search(distances, order=False, maxiter=np.inf):
    N = len(distances)
    if order is False:
        order = np.random.permutation(N)
    opt_dist = np.inf
    route_improved = True
    count = 0
    while route_improved is True and count < maxiter:
        count += 1
        route_improved = False
        for i in range(N):
            for j in range(i):
                if np.sum(order) != 276:
                    print(count, i, j, order)
                tmp = order[j]
                order[j] = order[i]
                order[i] = tmp
                d = travel_distance(distances, order)
                if d < opt_dist:
                    opt_dist = d
                    opt_order = order.copy()
                    route_improved = True
                else:  # change back rather than copying
                    tmp = order[j]
                    order[j] = order[i]
                    order[i] = tmp
    if np.sum(opt_order) != 276:
        print("c1", opt_order)
    return opt_order, opt_dist


def plot_solution(dist, order, latlon):
    with open('../doc/world_map.pkl', 'rb') as fid:
        ax_ = pickle.load(fid)
    xmin, xmax = -12, 40
    ymin, ymax = 35, 65
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    for i in range(len(latlon)):
        plt.text(latlon[i, 1], latlon[i, 0], str(i), fontsize=10)
    ax_.fill(latlon[order, 1], latlon[order, 0], edgecolor='b',
             fill=False, zorder=3)
    ax_.scatter(latlon[order, 1], latlon[order, 0],
                s=50, c="r", zorder=4)
    plt.title("traveling distance: {:.2f} km".format(dist))
    plt.show()
    return


def evolutution_TSP(distances, n_survivors, n_parents, n_generations,
                    parent_selection, recombination, survivor_selection,
                    hybrid=None):
    max_dist = np.sum(np.max(distances, axis=1))
    n_children = n_parents
    n_alleles = len(distances)
    children = np.empty((n_children, n_alleles), dtype=np.int32)
    survivors = np.empty((n_survivors, n_alleles), dtype=np.int32)
    population = np.empty((n_survivors+n_children, n_alleles), dtype=np.int32)
    fitness_chrn = np.empty((n_children,))
    fitness_surv = np.empty((n_survivors,))
    fitness_popu = np.empty((n_survivors+n_children,))
    shortest_dist = np.empty(n_generations,)
    # INITIALISE population with random candidate solution
    for i in range(n_survivors):
        survivors[i] = np.random.permutation(n_alleles)
        fitness_surv[i] = max_dist-travel_distance(distances, survivors[i])

    for generation in range(n_generations):
        # SELECT parents:
        parent_pairs = parent_selection(fitness_surv, n_parents)
        c = 0
        for parents in parent_pairs:
            gene_p1, gene_p2 = survivors[parents[0]], survivors[parents[1]]
            # RECOMBINE pairs of parents
            new_genes = recombination(gene_p1, gene_p2)
            # MUTATE the resulting offsprings
            child1, child2 = mutate(new_genes[0]), mutate(new_genes[1])
            if np.sum(child1) != 276:
                print("c1", child1)
            if np.sum(child2) != 276:
                print("c2", child2)
#            print(child1, child2)
            # EVALUATE new candidates
            if hybrid == "Lamarck":
                # In general the algorithms are referred to as Lamarckian if
                # the result of the local search stage replaces the individual
                # in the population.
                opt_child1, opt_dist1 = hillclimb_search(distances, child1, 1)
                opt_child2, opt_dist2 = hillclimb_search(distances, child2, 1)
                f1 = max_dist-travel_distance(distances, opt_child1)
                f2 = max_dist-travel_distance(distances, opt_child2)
                child1, child2 = opt_child1, opt_child2
            elif hybrid == "Baldwinian":
                # In general the algorithms are referred to as Baldwinian if
                # the original member is kept, but has as its fitness the value
                # belonging to the outcome of the local search process.
                opt_child1, opt_dist1 = hillclimb_search(distances, child1, 1)
                opt_child2, opt_dist2 = hillclimb_search(distances, child2, 1)
                f1 = max_dist-travel_distance(distances, opt_child1)
                f2 = max_dist-travel_distance(distances, opt_child2)
            else:
                f1 = max_dist-travel_distance(distances, child1)
                f2 = max_dist-travel_distance(distances, child2)
            children[c], children[c+1] = child1, child2
            fitness_chrn[c], fitness_chrn[c+1] = f1, f2
            c += 2
        # SELECT individuals for new generation
        survivor_selection(population, survivors, children,
                           fitness_popu, fitness_surv, fitness_chrn)
        best_candidate = survivors[np.argmax(fitness_surv)]
        shortest_dist[generation] = travel_distance(distances, best_candidate)

    if shortest_dist[-1] < 12287.0699:
        print(shortest_dist[-1])
        pprint(best_candidate)
        plot_solution(shortest_dist[-1], best_candidate, latlon)
    return shortest_dist


def test_exhaustive_search():
    print("exhaustive search:")
    for N in [4, 5, 6, 7, 8, 9, 10]:
        print(N, "cities", factorial(N), "options.")
        tic = time.perf_counter()
        opt_order, opt_dist = exhaustive_search(distances[:N, :N])
        toc = time.perf_counter()
        print("run time: ", toc - tic)
        print("shortest tour: ", opt_dist)
        print("city order: ", opt_order)
    plot_solution(opt_dist, opt_order, latlon[:N, :N])
    dt = toc-tic
    print("computation time for exhaustive seach of shortest tour between 24 cities: {:.0f} years".format(dt/60/60/24/365/factorial(10)*factorial(24)))
    return


def test_hillclimb_search():
    """write a simple hill climber to solve the TSP.
    How well does the hill climber perform, compared to the result from the
    exhaustive search for the first **10 cities**?
    Since you are dealing with a stochastic algorithm, you should run the
    algorithm several times to measure its performance.
    Report the length of the tour of the best, worst and mean of 20 runs
    (with random starting tours), as well as the standard deviation of the
    runs, both with the **10 first cities**, and with all **24 cities**."""

    tic = time.perf_counter()
    opt_order, opt_dist = hillclimb_search(distances[:10, :10])
    toc = time.perf_counter()
    t = toc-tic
    print("runtime: {:.6f} seconds. (= {:.2f} % compared to the exhaustive search)".format(t, t/25*100))
    tour_length = np.empty(20, )
    for n_cities in [10, 24]:
        print("number of cities:", n_cities, end="")
        for i in range(20):
            opt_order, opt_dist = hillclimb_search(distances[:n_cities, :n_cities])
            tour_length[i] = opt_dist
        print("""
        shortest tour: {:.2f} km
        longest tour: {:.2f} km
        average tour: {:.2f} km
        standard deviation {:.2f} km """.format(np.min(tour_length), np.max(tour_length),
                                                np.mean(tour_length), np.std(tour_length)))
    plot_solution(opt_dist, opt_order, latlon)
    return


def test_GA():
    """write a genetic algorithm (GA) to solve the problem.
    Choose mutation and crossover operators that are appropriate for the problem
    (see chapter 4.5 of the Eiben and Smith textbook).
    Choose three different values for the population size.
    Define and tune other parameters yourself and make assumptions as necessary
    (and report them, of course).

    For all three variants: As with the hill climber, report best, worst, mean
    and standard deviation of tour length out of 20 runs of the algorithm
    (of the best individual of last generation). Also, find and plot the
    average fitness of the best fit individual in each generation
    (average across runs), and include a figure with all three curves in
    the same plot in the report. Conclude which is best in terms of tour length
    and number of generations of evolution time.
    """
    n_generations = 15
    n_runs = 20
    tour_length = np.empty(20, )
    fig, ax = plt.subplots()
    b = np.empty((3, n_runs, n_generations))
    x = np.arange(n_generations)
    colors = ["r", "g", "b"]
    for i, n_survivors in enumerate([50, 100, 200]):
        c = colors[i]
        n_parents = n_survivors
        print("number of individuals:", n_survivors)
        print("run: ", end="")
        for j in range(n_runs):
            print(j, end=", ")
            b[i, j] = evolutution_TSP(distances, n_survivors, n_parents,
                                      n_generations, rank_based_selection,
                                      order_crossover_pair, elitism)
            tour_length[j] = b[i, j, -1]
        print("""
        shortest tour: {:.2f} km
        longest tour: {:.2f} km
        average tour: {:.2f} km
        standard deviation {:.2f} km """.format(np.min(tour_length), np.max(tour_length),
                                                np.mean(tour_length), np.std(tour_length)))
        y = np.mean(b[i], axis=0)
        plt.plot(x, y, c+"-", label="{:.0f} individuals".format(n_survivors))
    plt.legend()
    plt.xlim(0, n_generations)
    plt.xlabel("generation")
    plt.ylabel("shortest travel distance [km]")
    plt.title("evolution of the TSP solution (average of 20 runs)")
    plt.show()
    #plot_solution(opt_dist, opt_order, latlon)


def test_GA_recobination():
    n_generations = 250
    n_runs = 20
    fig, ax = plt.subplots()
    b = np.empty((3, n_runs, n_generations))
    x = np.arange(n_generations)
    recombinations = [pmx_pair, cycle_crossover_pair, order_crossover_pair]
    names = ["Partially Mapped Crossover", "Cycle Crossover", "Order Crossover"]
    colors = ["r", "g", "b"]
    for i in range(3):
        c = colors[i]
        recombination = recombinations[i]
        print("run: ", end="")
        for j in range(n_runs):
            print(j, end=", ")
            b[i, j] = evolutution_TSP(distances, 100, 100, n_generations,
                                   rank_based_selection, recombination, elitism)
        y = np.mean(b[i], axis=0)
        plt.plot(x, y, c+"-", label=names[i])
    plt.legend()
    plt.xlim(0, n_generations)
    plt.xlabel("generation")
    plt.ylabel("shortest travel distance [km]")
    plt.title("evolution of the TSP solution (average of 20 runs)")
    plt.show()
#    best = np.array([7, 11, 16, 3, 8, 6, 21, 19, 14, 10, 9, 4, 20, 1, 5, 23,
#                     22, 17, 2, 15, 13, 18, 0, 12])  # 12654
    best = np.array([17, 2, 23, 22, 5, 1, 20, 9, 4, 10, 14, 19, 21, 6, 8, 3,
                     16, 11, 7, 12, 0, 18, 13, 15])  # 12287.07
    d = travel_distance(distances, best)
    plot_solution(d, best, latlon)


def test_GA2():
    fig, (ax1, ax2) = plt.subplots(2)
    N = 10
    tic = time.perf_counter()
    tour_length = evolutution_TSP(distances[:N, :N], 100, 100,
                                  50, rank_based_selection,
                                  order_crossover_pair, elitism)
    toc = time.perf_counter()
    ax1.plot(np.arange(len(tour_length)), tour_length, "bo")
    ax1.set_title("evolution of the TSP solution ({:.0f} cities)".format(N))
    ax1.set_xlim(0, 50)
    print("run time: {:.3f} seconds".format(toc - tic))
    print("shortest tour: {:.2f} km".format(tour_length[-1]))
    print("shortest tour exhaustive search: {:.2f} km".format(7486.31))

    N = 24
    tic = time.perf_counter()
    tour_length = evolutution_TSP(distances[:N, :N], 500, 500,
                                  100, rank_based_selection,
                                  order_crossover_pair, elitism)
    toc = time.perf_counter()
    ax2.plot(np.arange(len(tour_length)), tour_length, "bo")
    ax2.set_title("evolution of the TSP solution ({:.0f} cities)".format(N))
    ax2.set_xlim(0, 100)
    print("run time: {:.3f} seconds".format(toc - tic))
    print("shortest tour: {:.2f} km".format(tour_length[-1]))
    print("shortest tour exhaustive search: {:.2f} km".format(12287.07))

    for ax in (ax1, ax2):
        ax.set_xlabel("generation")
        ax.set_ylabel("shortest travel distance [km]")
    fig.tight_layout(pad=2.0)

    case_10 = 100 * 50
    case_24 = 500 * 100
    print("10 cities: {:.0f} out of {:.0f} tours inspected (= {:.2f} %)".format(case_10, factorial(10), case_10/factorial(10)*100))
    print("24 cities: {:.0f} out of {:.0f} tours inspected (= {:.8f} %)".format(case_24, factorial(24), case_24/factorial(24)*100))


def test_hybrid(name):
    """Implement a hybrid algorithm to solve the TSP: Couple your GA and hill
    climber by running the hill climber a number of iterations on each
    individual in the population as part of the evaluation.
    Test both Lamarckian and Baldwinian learning models and report the results
    of both variants in the same way as with the pure GA (min, max, mean and
    standard deviation of the end result and an averaged generational plot).
    How do the results compare to that of the pure GA, considering the number
    of evaluations done?"""
    n_generations = 25
    n_runs = 20
    tour_length = np.empty(20, )
    fig, ax = plt.subplots()
    b = np.empty((3, n_runs, n_generations))
    x = np.arange(n_generations)
    colors = ["r", "g", "b"]
    for i, n_survivors in enumerate([10, 20, 30]):
        c = colors[i]
        n_parents = n_survivors
        print("number of individuals:", n_survivors)
        print("run: ", end="")
        for j in range(n_runs):
            print(j, end=", ")
            b[i, j] = evolutution_TSP(distances, n_survivors, n_parents,
                                      n_generations, rank_based_selection,
                                      order_crossover_pair, elitism,
                                      hybrid=name)  # Lamarck, Baldwinian
            tour_length[j] = b[i, j, -1]
        print("""
        shortest tour: {:.2f} km
        longest tour: {:.2f} km
        average tour: {:.2f} km
        standard deviation {:.2f} km """.format(np.min(tour_length), np.max(tour_length),
                                                np.mean(tour_length), np.std(tour_length)))
        y = np.mean(b[i], axis=0)
        plt.plot(x, y, c+"-", label="{:.0f} individuals".format(n_survivors))
    plt.legend()
    plt.xlim(0, n_generations)
    plt.xlabel("generation")
    plt.ylabel("shortest travel distance [km]")
    plt.title("evolution of the TSP solution (average of 20 runs)")
    plt.show()


if __name__ == "__main__":
    plt.close("all")
    distances = np.genfromtxt("../doc/european_cities.csv",
                              delimiter=";", skip_header=1)
    latlon = np.genfromtxt("../doc/latitude_lonitude.txt",
                           delimiter=";", skip_header=1)
#    test_exhaustive_search()
#    test_hillclimb_search()
#    test_GA_recobination()
#    test_GA()
#    test_GA2()
    test_hybrid("Lamarck")
    test_hybrid("Baldwinian") # Lamarck, Baldwinian
#    N = 9
#    fig, ax = plt.subplots()
#    exhaustive_search(distances[:N, :N],
#                      lambda i, d, o: plot_progress(ax, latlon, i, d, o))
#    plt.show()

#    N = 10
#    fig, ax = plt.subplots()
#    hillclimb_search(distances[:N, :N],
#                     lambda i, d, o: plot_progress(ax, latlon, i, d, o))
#    plt.show()
