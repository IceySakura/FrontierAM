"""Stable averaging of completed dyadic blocks of a digital net."""
import numpy as np


def query_uniform(oracle, points):
    values = oracle.ask(points)
    if len(values):
        if not hasattr(oracle, "uniform_samples"):
            oracle.uniform_samples = []
            oracle.uniform_count = 0
        oracle.uniform_samples.append(values.copy())
        oracle.uniform_count += len(values)
    return values


def uniform_values(oracle, fallback):
    samples = getattr(oracle, "uniform_samples", None)
    return np.concatenate(samples) if samples else np.asarray(fallback)


def mean(values):
    values = np.asarray(values)
    scale = np.max(np.abs(values))
    if scale == 0:
        return float(np.mean(values))
    return float(np.mean(values.astype(np.longdouble)/scale)*scale)


def standard_deviation(values):
    values = np.asarray(values)
    scale = np.max(np.abs(values))
    if scale == 0:
        return float(np.std(values))
    normalized = values.astype(np.longdouble)/scale
    return float(np.sqrt(np.mean((normalized-np.mean(normalized))**2))*scale)


def root_mean_square(values):
    values = np.asarray(values)
    scale = np.max(np.abs(values))
    if scale == 0:
        return float(np.mean(np.abs(values)))
    return float(np.sqrt(np.mean((values.astype(np.longdouble)/scale)**2))*scale)


def average(values, start=0):
    values = np.asarray(values)
    start = int(start)
    if len(values) == 0:
        raise ValueError("empty sample")
    scale = np.max(np.abs(values))
    if scale == 0:
        return float(np.mean(values))
    data = values.astype(np.longdouble)/scale
    total = np.longdouble(0)
    total_weight = np.longdouble(0)
    offset = 0
    while offset < len(data):
        size = 1 << ((len(data)-offset).bit_length()-1)
        index = start+offset
        if index:
            size = min(size, index & -index)
        weight = np.longdouble(size)*size
        total += weight*np.mean(data[offset:offset+size])
        total_weight += weight
        offset += size
    return float(total/total_weight*scale)
