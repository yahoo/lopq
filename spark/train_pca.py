# Copyright 2015, Yahoo Inc.
# Licensed under the terms of the Apache License, Version 2.0. See the LICENSE file associated with the project for terms.
from pyspark.context import SparkContext

import numpy as np
import base64
import cPickle as pkl
from tempfile import NamedTemporaryFile
import os
import subprocess
from operator import add


def default_data_loading(sc, data_path, sampling_ratio, seed):
    """
    This function loads training data from a text file, sampling it by the provided
    ratio and random seed, and interprets each line as a tab-separated (id, data) pair
    where 'data' is assumed to be a base64-encoded pickled numpy array. The ids are discarded.
    The data is returned as an RDD of numpy arrays.
    """
    # Compute the number of cores in our cluster - used below to heuristically set the number of partitions
    total_cores = int(sc._conf.get('spark.executor.instances')) * int(sc._conf.get('spark.executor.cores'))

    # Load and sample down the dataset
    d = sc.textFile(data_path, total_cores * 3).sample(False, sampling_ratio, seed)

    # The data is (id, vector) tab-delimited pairs where each vector is
    # a base64-encoded pickled numpy array
    deserialize_vec = lambda s: pkl.loads(base64.decodestring(s.split('\t')[1]))
    vecs = d.map(deserialize_vec)

    return vecs


def main(sc, args, data_load_fn=default_data_loading):

    def seqOp(a, b):
        a += np.outer(b, b)
        return a

    def combOp(a, b):
        a += b
        return a

    # Load data
    d = data_load_fn(sc, args.data, args.sampling_ratio, args.seed)
    d.cache()

    # Determine the data dimension
    D = len(d.first())

    # Count data points
    count = d.count()
    mu = d.aggregate(np.zeros(D), add, add)
    mu = mu / float(count)

    # Compute covariance estimator
    summed_covar = d.treeAggregate(np.zeros((D, D)), seqOp, combOp, depth=args.agg_depth)

    A = summed_covar / (count - 1) - np.outer(mu, mu)
    E, P = np.linalg.eigh(A)

    params = {
        'mu': mu,   # mean
        'P': P,     # PCA matrix
        'E': E,     # eigenvalues
        'A': A,     # covariance matrix
        'c': count  # sample size
    }

    save_hdfs_pickle(params, args.output)


def save_hdfs_pickle(m, pkl_path):
    """
    Given a python object and a path on hdfs, save the object as a pickle file locally and copy the file
    to the hdfs path.
    """
    print 'Saving pickle to temp file...'
    f = NamedTemporaryFile(delete=False)
    pkl.dump(m, f, -1)
    f.close()

    print 'Copying pickle file to hdfs...'
    copy_to_hdfs(f, pkl_path)
    os.remove(f.name)


def copy_to_hdfs(f, hdfs_path):
    subprocess.call(['hadoop', 'fs', '-copyFromLocal', f.name, hdfs_path])


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()

    # Data handling parameters
    parser.add_argument('--data', dest='data', type=str, required=True, help='hdfs path to input data')
    parser.add_argument('--data_udf', dest='data_udf', type=str, default=None, help='module name from which to load a data loading UDF')
    parser.add_argument('--seed', dest='seed', type=int, default=None, help='optional random seed')
    parser.add_argument('--sampling_ratio', dest='sampling_ratio', type=float, default=1.0, help='proportion of data to sample for training')
    parser.add_argument('--agg_depth', dest='agg_depth', type=int, default=4, help='depth of tree aggregation to compute covariance estimator')

    parser.add_argument('--output', dest='output', type=str, default=None, help='hdfs path to output pickle file of parameters')

    args = parser.parse_args()

    sc = SparkContext(appName='PCA')

    # Load UDF module if provided
    if args.data_udf:
        udf_module = __import__(args.data_udf, fromlist=['udf'])
        load_udf = udf_module.udf
        main(sc, args, data_load_fn=load_udf)
    else:
        main(sc, args)

    sc.stop()
