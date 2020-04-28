# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 16:31:27 2020

@author: Florian
"""
import numpy as np


def mutate(gene, f_scramble=0.01, f_swap=0.01, f_insert=0.01, f_inversion=0.01):
    """cause small, random variance to a genotype by shuffeling a random number
    of alleles.
    f_scramble: frequency of scrambled alleles.
        E.g. 0.3 -> 30% of the alleles are scrambeled
    f_swap: probability that one random swap of two alleles occurs.
    f_insert: probability that one random insert occurs.
    f_inversion: probability that one random inversion occurs between a random
        start and a random end.
    """
    mutated_gene = gene.copy()
    # insert
    if np.random.rand(1,) < f_insert:
        s1, s2 = np.random.randint(0, len(gene), 2)
        tmp = mutated_gene[s2]
        mutated_gene = np.delete(mutated_gene, s2)
        mutated_gene = np.insert(mutated_gene, s1, tmp)
    # SWAP
    if np.random.rand(1,) < f_swap:
        s1, s2 = np.random.randint(0, len(gene), 2)
        tmp = mutated_gene[s1]
        mutated_gene[s1] = mutated_gene[s2]
        mutated_gene[s2] = tmp
    # scramble
    mutate = np.random.rand(*gene.shape) < f_scramble
    mutations = mutated_gene[mutate]
    np.random.shuffle(mutations)
    mutated_gene[mutate] = mutations
    # inversion
    if np.random.rand(1,) < f_inversion:
        s, e = np.random.randint(0, len(gene), 2)
        if s > e: s, e = e, s
        mutated_gene[s:e] = mutated_gene[s:e][::-1]
#    if np.sum(mutated_gene) != 276:
#        print(gene, mutated_gene)
    return mutated_gene


def pmx(a, b):
    s, e = np.random.randint(0, len(a), 2)
    if s > e: s, e = e, s
    child = np.empty(a.shape, dtype=np.int32)
    child[s:e] = a[s:e]
    s1 = a[s:e]
    s2 = b[s:e]
    mapping = dict(zip(s1, s2))  # s1 are the keys, s2 the values
    for i in range(len(a)):
        if not ((s <= i) & (i < e)):
            new_allel = b[i]
            while new_allel in mapping.keys():
                new_allel = mapping[new_allel]
            child[i] = new_allel
    return child


def pmx_pair(p1, p2):
    a = np.array(p1)
    b = np.array(p2)
    children = np.empty((2, len(p1)), dtype=np.int32)
    children[0] = pmx(a, b)
    children[1] = pmx(b, a)
    return children


def order_crossover(a, b):
    """Order crossover"""
    s, e = np.random.randint(0, len(a), 2)
    if s > e: s, e = e, s
    s1 = a[s:e]
    child = np.empty(len(b), dtype=np.int32)
    child[s:e] = s1
    pos1 = e
    pos2 = e
    next_num = b[pos2]
    while pos1 != s:
        while next_num in s1:
            pos2 += 1
            if pos2 == len(a):
                pos2 = 0
            next_num = b[pos2]
        child[pos1] = next_num
        pos1 += 1
        pos2 += 1
        if pos1 == len(a):
            pos1 = 0
        if pos2 == len(a):
            pos2 = 0
        next_num = b[pos2]
    return child


def order_crossover_pair(p1, p2):
    a = np.array(p1)
    b = np.array(p2)
    children = np.empty((2, len(p1)), dtype=np.int32)
    children[0] = order_crossover(a, b)
    children[1] = order_crossover(b, a)
    return children


def cycle_crossover(a, b):
    """https://www.youtube.com/watch?v=DJ-yBmEEkgA"""
    # what if there is only one cycle?
    # what if there are three or ore cycles?
    # implementation: -> each child is a copy of parent 1
    #                    + one cycle from parent 2
    # ! THERE MIGHT BE MORE THAN ONE CHILD !!!
    children = []
    first_points = np.arange(len(a))
    while len(first_points) > 0:
        indxs_c0 = []
        cycle_incomplete = True
        i0 = first_points[0]
        i = i0
        while cycle_incomplete:
            v = b[i]
            for j in range(len(a)):
                if a[j] == v:
                    i = j
                    break
            indxs_c0.append(i)
            l = first_points != i
            first_points = first_points[l]
            if i == i0:
                cycle_incomplete = False
        if len(indxs_c0)>1:
            child0 = a.copy()
            child0[indxs_c0] = b[indxs_c0]
            children.append(child0)
    return children


def cycle_crossover_pair(p1, p2):
    a = np.array(p1)
    b = np.array(p2)
    children = []
    children1 = cycle_crossover(a, b)
    children2 = cycle_crossover(b, a)
    children = children1 + children2

    N = len(children)
    if N > 2:
#        print("!!! got more than 2 children !!!", N)
#        print()
#        print(p1)
#        print(p2)
#        print()
#        for c in children:
#            print(c, end = " : ")
#            for d in children:
#                print(sum(c==d), end=" ")
#            print()
        c1 = children[np.random.randint(0, high=N)]
        c2 = children[np.random.randint(0, high=N)]
        children = [c1, c2]
    if N == 0:
        children = [p1, p2]
    return children
