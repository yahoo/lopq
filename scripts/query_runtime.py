# Copyright 2015, Yahoo Inc.
# Licensed under the terms of the Apache License, Version 2.0. See the LICENSE file associated with the project for terms.
from math import sqrt, ceil


def multiseq_flops(V, D):
    """
    Given the number of coarse clusters and the dimension of the data,
    compute the number of flops to required to rank each coarse vocabulary
    to the query.
    """
    # (total coarse vocabulary) * (dims per coarse split) * (flops per squared distance)
    return (2 * V) * (D / 2) * 2


def cluster_rotation_flops(D):
    """
    Given the dimension of the data, compute the number of flops for a
    single local projection.
    """
    D2 = D / 2
    return D2 ** 2 + D2


def subquantizer_flops(D, M, clusters=256):
    """
    Given the dimension of the data, the number of subquantizers and the
    subquantizer vocabulary size, compute the number of flops to compute
    a projected query's LOPQ distance for a single half of the query.
    """
    # (subquants per half) * (dims per subquant) * (cluster per subquant) * (flops per squared distance)
    return (M / 2) * (D / M) * clusters * 2


def total_rank_flops(D, M, N, cells, badness=0.5):
    """
    Given the dimension of the data, the number of subquantizers, the number
    of results to rank, the number of multi-index cells retrieved by the query,
    and a badness measure that interpolates between best case and worst case
    in terms of reusable (cacheable) subquantizer distance computations, compute
    the number of flops to rank the N results.

    The 'badness' will vary query to query and is determined by the data distribution.
    """
    # Corresponds to traversing a row or column of the multi-index grid
    worst_case = cells + 1

    # Corresponds to traversing a square in the multi-index grid
    best_case = 2 * sqrt(cells)

    # Interpolated number of clusters
    num_clusters = ceil(worst_case * badness + best_case * (1 - badness))

    # (total local projections required) + (total number of sums to compute distance)
    return num_clusters * (cluster_rotation_flops(D) + subquantizer_flops(D, M)) + N * (M - 1)


def brute_force_flops(D, N):
    """
    Given the data dimension and the number of results to rank, compute
    the number of flops for brute force exact distance computation.
    """
    return N * (3 * D)


def ratio(D, M, N, cells, badness=0.5):
    return total_rank_flops(D, M, N, cells, badness) / brute_force_flops(D, N)
