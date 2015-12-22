# Copyright 2015, Yahoo Inc.
# Licensed under the terms of the Apache License, Version 2.0. See the LICENSE file associated with the project for terms.
import heapq
from collections import defaultdict, namedtuple
import numpy as np
from .utils import iterate_splits, parmap, get_chunk_ranges


def multisequence(x, centroids):
    """
    Implementation of multi-sequence algorithm for traversing a multi-index.

    The algorithm is described in http://download.yandex.ru/company/cvpr2012.pdf.

    :param ndarray x:
        a query vector
    :param list centroids:
        a list of ndarrays containing cluster centroids for each subvector

    :yields int d:
        the cell distance approximation used to order cells
    :yields tuple cell:
        the cell indices
    """

    # Infer parameters
    splits = len(centroids)
    V = centroids[0].shape[0]

    # Compute distances to each coarse cluster and sort
    cluster_dists = []
    sorted_inds = []
    for cx, split in iterate_splits(x, splits):

        dists = ((cx - centroids[split]) ** 2).sum(axis=1)
        inds = np.argsort(dists)

        cluster_dists.append(dists)
        sorted_inds.append(inds)

    # Some helper functions used below
    def cell_for_inds(inds):
        return tuple([sorted_inds[s][i] for s, i in enumerate(inds)])

    def dist_for_cell(cell):
        return sum([cluster_dists[s][i] for s, i in enumerate(cell)])

    def inds_in_range(inds):
        for i in inds:
            if i >= V:
                return False
        return True

    # Initialize priority queue
    h = []
    traversed = set()
    start_inds = tuple(0 for _ in xrange(splits))
    start_dist = dist_for_cell(cell_for_inds(start_inds))
    heapq.heappush(h, (start_dist, start_inds))

    # Traverse cells
    while len(h):
        d, inds = heapq.heappop(h)
        yield d, cell_for_inds(inds)
        traversed.add(inds)

        # Add neighboring cells to queue
        if inds[1] == 0 or (inds[0] + 1, inds[1] - 1) in traversed:
            c = (inds[0] + 1, inds[1])
            if inds_in_range(c):
                dist = dist_for_cell(cell_for_inds(c))
                heapq.heappush(h, (dist, c))

        if inds[0] == 0 or (inds[0] - 1, inds[1] + 1) in traversed:
            c = (inds[0], inds[1] + 1)
            if inds_in_range(c):
                dist = dist_for_cell(cell_for_inds(c))
                heapq.heappush(h, (dist, c))


class LOPQSearcher(object):
    def __init__(self, model):
        """
        Create an LOPQSearcher instance that encapsulates retrieving and ranking
        with LOPQ. Requires an LOPQModel instance.
        """
        self.model = model
        self.index = defaultdict(list)

    def add_data(self, data, ids=None, num_procs=1):
        """
        Add raw data into the search index.

        :param ndarray data:
            an ndarray with data points on the rows
        :param ndarray ids:
            an optional array of ids for each data point;
            defaults to the index of the data point if not provided
        :param int num_procs:
            an integer specifying the number of processes to use to
            compute codes for the data
        """
        N = data.shape[0]

        # If a list of ids is not provided, assume it is the index of the data
        if ids is None:
            ids = range(N)

        # function to index a partition of the data
        def index_partition(work):
            from collections import defaultdict

            data, ids = work
            index = defaultdict(list)
            for item_id, d in zip(ids, data):
                code = self.model.predict(d)
                cell = code[0]
                index[cell].append((item_id, code))
            return index

        def merge_dicts(a, b):
            for k, v in b.iteritems():
                a[k] += v
            return a

        if num_procs > 1:
            tasks = [(data[a:b], ids[a:b]) for a, b in get_chunk_ranges(N, num_procs)]
            index_dicts = parmap(index_partition, tasks, num_procs)
        else:
            index_dicts = map(index_partition, [(data, ids)])

        self.index = reduce(merge_dicts, index_dicts, self.index)

    def get_result_quota(self, x, quota=10):
        """
        Given a query vector and result quota, retrieve as many cells as necessary
        to fill the quota.

        :param ndarray x:
            a query vector
        :param int quota:
            the desired number of items to retrieve

        :returns list retrieved:
            a list of index items
        :returns int visited:
            the number of multi-index cells visited
        """
        retrieved = []
        visited = 0
        for _, cell in multisequence(x, self.model.Cs):
            if cell not in self.index:
                continue

            retrieved += self.index[cell]
            visited += 1

            if len(retrieved) >= quota:
                break

        return retrieved, visited

    def compute_distances(self, x, items):
        """
        Given a query and a list of index items, compute the approximate distance of the query
        to each item and return a list of tuples that contain the distance and the item.
        Memoize subquantizer distances per coarse cluster to save work.

        :param ndarray x:
            a query vector
        :param list items:
            a list of items from the index

        :returns list:
            a list of items with distance
        """
        memoized_subquant_dists = [{}, {}]

        def get_subquantizer_distances(x, coarse):

            d0, d1 = memoized_subquant_dists
            c0, c1 = coarse

            if c0 not in d0:
                d0[c0] = self.model.get_subquantizer_distances(x, coarse, coarse_split=0)

            if c1 not in d1:
                d1[c1] = self.model.get_subquantizer_distances(x, coarse, coarse_split=1)

            return d0[c0] + d1[c1]

        results = []
        for item in items:

            codes = item[1]
            coarse, fine = codes

            subquantizer_distances = get_subquantizer_distances(x, coarse)
            dist = sum([subquantizer_distances[i][fc] for i, fc in enumerate(fine)])

            results.append((dist, item))

        return results

    def search(self, x, quota=10, with_dists=False):
        """
        Return euclidean distance ranked results, along with the number of cells
        traversed to fill the quota.

        :param ndarray x:
            a query vector
        :param int quota:
            the number of desired results
        :param bool with_dists:
            boolean indicating whether result items should be returned with their distance

        :returns list results:
            the list of ranked results
        :returns int visited:
            the number of cells visited in the query
        """
        # Retrieve results with multi-index
        retrieved, visited = self.get_result_quota(x, quota)

        # Compute distance for results
        results = self.compute_distances(x, retrieved)

        # Sort by distance
        results = sorted(results, key=lambda d: d[0])

        if with_dists:
            Result = namedtuple('Result', ['id', 'code', 'dist'])
            results = map(lambda d: Result(d[1][0], d[1][1], d[0]), results)
        else:
            Result = namedtuple('Result', ['id', 'code'])
            results = map(lambda d: Result(d[1][0], d[1]), results)

        return results, visited
