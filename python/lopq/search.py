# Copyright 2015, Yahoo Inc.
# Licensed under the terms of the Apache License, Version 2.0. See the LICENSE file associated with the project for terms.
import heapq
from collections import defaultdict, namedtuple
from itertools import count
import numpy as np
import array
from .utils import iterate_splits, compute_codes_parallel


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


class LOPQSearcherBase(object):

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
        codes = compute_codes_parallel(data, self.model, num_procs)
        self.add_codes(codes, ids)

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
            retrieved += self.get_cell(cell)
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

    def search(self, x, quota=10, limit=None, with_dists=False):
        """
        Return euclidean distance ranked results, along with the number of cells
        traversed to fill the quota.

        :param ndarray x:
            a query vector
        :param int quota:
            the number of desired results to rank
        :param int limit:
            the number of desired results to return - defaults to quota
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

        # Limit number returned
        if limit is None:
            limit = quota
        results = results[:limit]

        if with_dists:
            Result = namedtuple('Result', ['id', 'code', 'dist'])
            results = map(lambda d: Result(d[1][0], d[1][1], d[0]), results)
        else:
            Result = namedtuple('Result', ['id', 'code'])
            results = map(lambda d: Result(d[1][0], d[1]), results)

        return results, visited

    def add_codes(self, codes, ids=None):
        """
        Add LOPQ codes into the search index.

        :param iterable codes:
            an iterable of LOPQ code tuples
        :param iterable ids:
            an optional iterable of ids for each code;
            defaults to the index of the code tuple if not provided
        """
        raise NotImplementedError()

    def get_cell(self, cell):
        """
        Retrieve a cell bucket from the index.

        :param tuple cell:
            a cell tuple

        :returns list:
            the list of index items in this cell bucket
        """
        raise NotImplementedError()

class LOPQSearcher(LOPQSearcherBase):
    def __init__(self, model):
        """
        Create an LOPQSearcher instance that encapsulates retrieving and ranking
        with LOPQ. Requires an LOPQModel instance. This class uses a Python dict
        to implement the index.

        :param LOPQModel model:
            the model for indexing and ranking
        """
        self.model = model
        self.index = defaultdict(list)

    def add_codes(self, codes, ids=None):
        """
        Add LOPQ codes into the search index.

        :param iterable codes:
            an iterable of LOPQ code tuples
        :param iterable ids:
            an optional iterable of ids for each code;
            defaults to the index of the code tuple if not provided
        """
        # If a list of ids is not provided, assume it is the index of the data
        if ids is None:
            ids = count()

        for item_id, code in zip(ids, codes):
            cell = code[0]
            self.index[cell].append((item_id, code))

    def get_cell(self, cell):
        """
        Retrieve a cell bucket from the index.

        :param tuple cell:
            a cell tuple

        :returns list:
            the list of index items in this cell bucket
        """
        return self.index[cell]


class LOPQSearcherLMDB(LOPQSearcherBase):
    def __init__(self, model, lmdb_path, id_lambda=int):
        """
        Create an LOPQSearcher instance that encapsulates retrieving and ranking
        with LOPQ. Requires an LOPQModel instance. This class uses an lmbd database
        to implement the index.

        :param LOPQModel model:
            the model for indexing and ranking
        :param str lmdb_path:
            path for the lmdb database; if it does not exist it is created
        :param callable id_lambda:
            a lambda function to reconstruct item ids from their string representation
            (computed by calling `bytes`) during retrieval
        """
        import lmdb

        self.model = model
        self.lmdb_path = lmdb_path
        self.id_lambda = id_lambda

        self.env = lmdb.open(self.lmdb_path, map_size=1024*2000000*2, writemap=False, map_async=True, max_dbs=1)
        self.index_db = self.env.open_db("index")

    def encode_cell(self, cell):
        return array.array("H", cell).tostring()

    def decode_cell(self, cell_bytes):
        a = array.array("H")
        a.fromstring(cell_bytes)
        return tuple(a.tolist())

    def encode_fine_codes(self, fine):
        return array.array("B", fine).tostring()

    def decode_fine_codes(self, fine_bytes):
        a = array.array("B")
        a.fromstring(fine_bytes)
        return tuple(a.tolist())

    def add_codes(self, codes, ids=None):
        """
        Add LOPQ codes into the search index.

        :param iterable codes:
            an iterable of LOPQ code tuples
        :param iterable ids:
            an optional iterable of ids for each code;
            defaults to the index of the code tuple if not provided
        """
        # If a list of ids is not provided, assume it is the index of the data
        if ids is None:
            ids = count()

        with self.env.begin(db=self.index_db, write=True) as txn:
            for item_id, code in zip(ids, codes):
                key_prefix = self.encode_cell(code[0])
                key_suffix = bytes(item_id)
                key = key_prefix + key_suffix
                val = self.encode_fine_codes(code[1])
                txn.put(key, val)
        self.env.sync()

    def get_cell(self, cell):
        """
        Retrieve a cell bucket from the index.

        :param tuple cell:
            a cell tuple

        :returns list:
            the list of index items in this cell bucket
        """
        prefix = self.encode_cell(cell)

        items = []
        with self.env.begin(db=self.index_db) as txn:
            cursor = txn.cursor()
            cursor.set_range(prefix)
            for key, value in cursor:
                if not key.startswith(prefix):
                    break
                else:
                    item_id = self.id_lambda(key[4:])
                    cell = self.decode_cell(key[:4])
                    fine = self.decode_fine_codes(value)
                    code = (cell, fine)
                    items.append((item_id, code))
            cursor.close()

        return items
