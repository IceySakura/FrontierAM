"""Conservative memory-order selection for finite-valued observations."""

import numpy as np


def log_likelihood(counts):
    totals = counts.sum(axis=1)
    probability = counts/np.maximum(totals[:, None], 1)
    positive = counts > 0
    value = np.sum(counts[positive]*np.log(probability[positive]))
    parameters = np.count_nonzero(positive)-np.count_nonzero(totals)
    return value, parameters


def connected(counts):
    graph = counts > 0
    for adjacency in (graph, graph.T):
        reached = np.zeros(len(graph), dtype=bool)
        reached[0] = True
        for _ in range(len(graph)):
            updated = reached | adjacency[reached].any(axis=0)
            if np.array_equal(reached, updated):
                break
            reached = updated
        if not reached.all():
            return False
    return True


def second_order(states, labels, counts):
    n, k = len(labels), len(states)
    pairs = labels[:-1]*k+labels[1:]
    keys, inverse, frequency = np.unique(pairs, return_inverse=True, return_counts=True)
    size = len(keys)
    if size <= k or size > min(512, (n-1)//12):
        return None
    transitions = np.bincount(
        inverse[:-1]*size+inverse[1:], minlength=size*size
    ).reshape(size, size).astype(float)
    parent = np.bincount(
        labels[1:-1]*k+labels[2:], minlength=k*k
    ).reshape(k, k).astype(float)
    full_likelihood, full_parameters = log_likelihood(transitions)
    parent_likelihood, parent_parameters = log_likelihood(parent)
    extra = max(full_parameters-parent_parameters, size-k)
    if 2*(full_likelihood-parent_likelihood) <= np.log(n-2)*extra:
        return None
    if not connected(counts):
        return None
    parent_probability = counts/counts.sum(axis=1)[:, None]
    prior = np.zeros((size, size))
    previous, current = keys//k, keys % k
    for i, state in enumerate(current):
        available = np.flatnonzero(previous == state)
        prior[i, available] = parent_probability[state, current[available]]
    prior /= prior.sum(axis=1)[:, None]
    return states[current], inverse, frequency, transitions, prior
