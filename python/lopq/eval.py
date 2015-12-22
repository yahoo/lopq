# Copyright 2015, Yahoo Inc.
# Licensed under the terms of the Apache License, Version 2.0. See the LICENSE file associated with the project for terms.
import time
import numpy as np


def compute_all_neighbors(data1, data2=None, just_nn=True):
    """
    For each point in data1, compute a ranked list of neighbor indices from data2.
    If data2 is not provided, compute neighbors relative to data1

    :param ndarray data1:
        an m1 x n dim matrix with observations on the rows
    :param ndarray data2:
        an m2 x n dim matrix with observations on the rows

    :returns ndarray:
        an m1 x m2 dim matrix with the distance-sorted indices of neighbors on the rows
    """
    from scipy.spatial.distance import cdist

    if data2 is None:
        data2 = data1

    dists = cdist(data1, data2)

    if just_nn:
        nns = np.zeros((data1.shape[0]), dtype=int)
    else:
        nns = np.zeros(dists.shape, dtype=int)

    for i in xrange(dists.shape[0]):
        if just_nn:
            nns[i] = np.argmin(dists[i])
        else:
            nns[i] = np.argsort(dists[i])

    return nns


def get_proportion_nns_with_same_coarse_codes(data, model, nns=None):
    """
    """
    N = data.shape[0]

    # Compute nearest neighbors if not provided
    if nns is None:
        nns = compute_all_neighbors(data)

    # Compute coarse codes for data
    coarse_codes = []
    for d in data:
        c = model.predict_coarse(d)
        coarse_codes.append(c)

    # Count the number of NNs that share the same coarse codes
    count = 0
    for i in xrange(N):
        nn = nns[i]
        if coarse_codes[i] == coarse_codes[nn]:
            count += 1

    return float(count) / N


def get_cell_histogram(data, model):
    # Compute cells for data
    cells = []
    for d in data:
        c = model.predict_coarse(d)
        cell = model.get_cell_id_for_coarse_codes(c)
        cells.append(cell)

    return np.histogram(cells, bins=range(model.V ** 2))[0]


def get_proportion_of_reconstructions_with_same_codes(data, model):
    N = data.shape[0]

    # Compute coarse codes for data
    count = 0
    for d in data:
        c1 = model.predict(d)
        r = model.reconstruct(c1)
        c2 = model.predict(r)
        if c1 == c2:
            count += 1

    return float(count) / N


def get_recall(searcher, queries, nns, thresholds=[1, 10, 100, 1000], normalize=True, verbose=False):
    """
    Given a LOPQSearcher object with indexed data and groundtruth nearest neighbors for a set of test
    query vectors, collect and return recall statistics.

    :param LOPQSearcher searcher:
        a searcher that contains the indexed nearest neighbors
    :param ndarray queries:
        a collect of test vectors with vectors on the rows
    :param ndarray nns:
        a list of true nearest neighbor ids for each vector in queries
    :param list thresholds:
        the recall thresholds to evaluate - the last entry defines the number of
        results to retrieve before ranking
    :param bool normalize:
        flag to indicate whether the result should be normalized by the number of queries
    :param bool verbose:
        flag to print every 50th search to visualize progress

    :return list:
        a list of recalls for each specified threshold level
    :return float:
        the elapsed query time
    """

    recall = np.zeros(len(thresholds))
    query_time = 0.0
    for i, d in enumerate(queries):

        nn = nns[i]

        start = time.clock()
        results, cells_visited = searcher.search(d, thresholds[-1])
        query_time += time.clock() - start

        if verbose and i % 50 == 0:
            print '%d cells visitied for query %d' % (cells_visited, i)

        for j, res in enumerate(results):
            rid, code = res

            if rid == nn:
                for k, t in enumerate(thresholds):
                    if j < t:
                        recall[k] += 1

    if normalize:
        N = queries.shape[0]
        return recall / N, query_time / N
    else:
        return recall, query_time


def get_subquantizer_distortion(data, model):
    from .model import compute_residuals, project_residuals_to_local

    first_half, second_half = np.split(data, 2, axis=1)

    r1, a1 = compute_residuals(first_half, model.Cs[0])
    r2, a2 = compute_residuals(second_half, model.Cs[1])

    p1 = project_residuals_to_local(r1, a1, model.Rs[0], model.mus[0])
    p2 = project_residuals_to_local(r2, a2, model.Rs[1], model.mus[1])

    pall = np.concatenate((p1, p2), axis=1)
    suball = model.subquantizers[0] + model.subquantizers[1]

    dists = np.array([sum(np.linalg.norm(compute_residuals(d, c)[0], ord=2, axis=1) ** 2) for c, d in zip(suball, np.split(pall, 8, axis=1))])

    return dists / data.shape[0]
