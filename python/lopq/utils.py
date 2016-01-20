# Copyright 2015, Yahoo Inc.
# Licensed under the terms of the Apache License, Version 2.0. See the LICENSE file associated with the project for terms.
import numpy as np
import multiprocessing
from itertools import chain


def iterate_splits(x, splits):
    """
    A helper to iterate subvectors.

    :param ndarray x:
        a vector to iterate over
    :param int splits:
        the number of subvectors
    :returns (np.array, int):
        subvector, split index pairs
    """
    split_size = len(x) / splits
    for split in xrange(splits):
        start = split * split_size
        yield x[start:start + split_size], split


def concat_new_first(arrs):
    """
    Helper to concatenate a list of ndarrays along a new first dimension.
    """
    arrs = map(lambda x: x[np.newaxis, ...], arrs)
    return np.concatenate(arrs, axis=0)


def predict_cluster(x, centroids):
    """
    Given a vector of dimension D and a matrix of centroids of dimension VxD,
    return the id of the closest cluster

    :params np.array x:
        the data to assign
    :params np.array centroids:
        a matrix of cluster centroids
    :returns int:
        cluster assignment
    """
    return ((x - centroids) ** 2).sum(axis=1).argmin(axis=0)


def load_xvecs(filename, base_type='f', max_num=None):
    """
    A helper to read in sift1m binary dataset. This parses the
    binary format described at http://corpus-texmex.irisa.fr/.

    :returns ndarray:
        a N x D array, where N is the number of observations
        and D is the number of features
    """
    import os
    import struct

    format_code, format_size, py_type = {
        'f': ('f', 4, float),
        'i': ('I', 4, int),
        'b': ('B', 1, float)
    }[base_type]

    size = os.path.getsize(filename)

    f = open(filename, 'rb')
    D = np.uint32(struct.unpack('I', f.read(4))[0])
    N = size / (4 + D * format_size)

    if max_num is None:
        max_num = N

    f.seek(0)
    A = np.zeros((max_num, D), dtype=py_type)
    for i in xrange(max_num):
        for j in xrange(D + 1):
            if j == 0:
                np.uint32(struct.unpack(format_code, f.read(4)))
            else:
                A[i, j - 1] = py_type(struct.unpack(format_code, f.read(format_size))[0])
    f.close()
    return np.squeeze(A)


def save_xvecs(data, filename, base_type='f'):
    """
    A helper to save an ndarray in the binary format as is expected in
    load_xvecs above.
    """
    import struct

    format_code, format_size, py_type = {
        'f': ('f', 4, float),
        'i': ('I', 4, int),
        'b': ('B', 1, float)
    }[base_type]

    f = open(filename, 'wb')
    for d in data:

        if hasattr(d, "__len__"):
            D = len(d)

            f.write(struct.pack('<I', D))
            for x in d:
                f.write(struct.pack(format_code, x))
        else:
            D = 1
            f.write(struct.pack('<I', D))
            f.write(struct.pack(format_code, d))

    f.flush()
    f.close()


def parmap(f, X, nprocs=multiprocessing.cpu_count()):
    """
    Parallel map implementation adapted from http://stackoverflow.com/questions/3288595/multiprocessing-using-pool-map-on-a-function-defined-in-a-class
    """

    def func_wrap(f, q_in, q_out):
        while True:
            i, x = q_in.get()
            if i is None:
                break
            q_out.put((i, f(x)))

    q_in = multiprocessing.Queue(1)
    q_out = multiprocessing.Queue()

    proc = [multiprocessing.Process(target=func_wrap, args=(f, q_in, q_out)) for _ in range(nprocs)]
    for p in proc:
        p.daemon = True
        p.start()

    sent = [q_in.put((i, x)) for i, x in enumerate(X)]
    [q_in.put((None, None)) for _ in range(nprocs)]
    res = [q_out.get() for _ in range(len(sent))]

    [p.join() for p in proc]
    [p.terminate() for p in proc]

    return [x for i, x in sorted(res)]


def get_chunk_ranges(N, num_procs):
    """
    A helper that given a number N representing the size of an iterable and the num_procs over which to
    divide the data return a list of (start_index, end_index) pairs that divide the data as evenly as possible
    into num_procs buckets.
    """
    per_thread = N / num_procs
    allocation = [per_thread] * num_procs
    allocation[0] += N - num_procs * per_thread
    data_ranges = [0] + reduce(lambda acc, num: acc + [num + (acc[-1] if len(acc) else 0)], allocation, [])
    data_ranges = [(data_ranges[i], data_ranges[i + 1]) for i in range(len(data_ranges) - 1)]
    return data_ranges


def compute_codes_parallel(data, model, num_procs=4):
    """
    A helper function that parallelizes the computation of LOPQ codes in 
    a configurable number of processes.

    :param ndarray data:
        an ndarray of data points
    :param LOPQModel model:
        a model instance to use to compute codes
    :param int num_procs:
        the number of processes to spawn

    :returns iterable:
        an iterable of computed codes in the input order
    """
    def compute_partition(data):
        return [model.predict(d) for d in data]

    N = len(data)
    partitions = [data[a:b] for a, b in get_chunk_ranges(N, num_procs)]
    codes = parmap(compute_partition, partitions, num_procs)

    return chain(*codes)
