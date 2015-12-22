# Copyright 2015, Yahoo Inc.
# Licensed under the terms of the Apache License, Version 2.0. See the LICENSE file associated with the project for terms.
from pyspark.context import SparkContext

import numpy as np
import cPickle as pkl
import base64
import os
import subprocess
import sys
from tempfile import NamedTemporaryFile
from operator import add

from pyspark.mllib.clustering import KMeans, KMeansModel
from lopq.model import LOPQModel, compute_rotations_from_accumulators


STEP_COARSE = 0
STEP_ROTATION = 1
STEP_SUBQUANT = 2


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


def load_data(sc, args, data_load_fn=default_data_loading):
    """
    Load training data as an RDD.
    """
    # Load data
    vecs = data_load_fn(sc, args.data, args.sampling_ratio, args.seed)

    # Split the vectors
    split_vecs = vecs.map(lambda x: np.split(x, 2))

    return split_vecs


def train_coarse(sc, split_vecs, V, seed=None):
    """
    Perform KMeans on each split of the data with V clusters each.
    """

    # Cluster first split
    first = split_vecs.map(lambda x: x[0])
    first.cache()
    print 'Total training set size: %d' % first.count()
    print 'Starting training coarse quantizer...'
    C0 = KMeans.train(first, V, initializationMode='random', maxIterations=10, seed=seed)
    print '... done training coarse quantizer.'
    first.unpersist()

    # Cluster second split
    second = split_vecs.map(lambda x: x[1])
    second.cache()
    print 'Starting training coarse quantizer...'
    C1 = KMeans.train(second, V, initializationMode='random', maxIterations=10, seed=seed)
    print '... done training coarse quantizer.'
    second.unpersist()

    return np.vstack(C0.clusterCenters), np.vstack(C1.clusterCenters)


def train_rotations(sc, split_vecs, M, Cs):
    """
    For compute rotations for each split of the data using given coarse quantizers.
    """

    Rs = []
    mus = []
    counts = []
    for split in xrange(2):

        print 'Starting rotation fitting for split %d' % split

        # Get the data for this split
        data = split_vecs.map(lambda x: x[split])

        # Get kmeans model
        model = KMeansModel(Cs[split])

        R, mu, count = compute_local_rotations(sc, data, model, M / 2)
        Rs.append(R)
        mus.append(mu)
        counts.append(count)

    return Rs, mus, counts


def accumulate_covariance_estimators(sc, data, model):
    """
    Analogous function to function of the same name in lopq.model.

    :param SparkContext sc:
        a SparkContext
    :param RDD data:
        an RDD of numpy arrays
    :param KMeansModel model:
        a KMeansModel instance for which to fit local rotations
    """

    def get_residual(x):
        cluster = model.predict(x)
        centroid = model.clusterCenters[cluster]
        residual = x - centroid
        return (cluster, residual)

    def seq_op(acc, x):
        acc += np.outer(x, x)
        return acc

    # Compute (assignment, residual) k/v pairs
    residuals = data.map(get_residual)
    residuals.cache()

    # Collect counts and mean residuals
    count = residuals.countByKey()
    mu = residuals.reduceByKey(add).collectAsMap()

    # Extract the dimension of the data
    D = len(mu.values()[0])

    # Collect accumulated outer products
    A = residuals.aggregateByKey(np.zeros((D, D)), seq_op, add).collectAsMap()

    residuals.unpersist()

    return A, mu, count


def dict_to_ndarray(d, N):
    """
    Helper for collating a dict with int keys into an ndarray. The value for a key
    becomes the value at the corresponding index in the ndarray and indices missing
    from the dict become zero ndarrays of the same dimension.

    :param dict d:
        a dict of (int, ndarray) or (int, number) key/values
    :param int N:
        the size of the first dimension of the new ndarray (the rest of the dimensions
        are determined by the shape of elements in d)
    """

    el = d.values()[0]
    if type(el) == np.ndarray:
        value_shape = el.shape
        arr = np.zeros((N,) + value_shape)
    else:
        arr = np.zeros(N)

    for i in d:
        arr[i] = d[i]
    return arr


def compute_local_rotations(sc, data, model, num_buckets):
    """
    Analogous to the function of the same name in lopq.model.

    :param SparkContext sc:
        a SparkContext
    :param RDD data:
        an RDD of numpy arrays
    :param KMeansModel model:
        a KMeansModel instance for which to fit local rotations
    :param int num_buckets:
        the number of subvectors over which to balance residual variance
    """
    # Get estimators
    A, mu, count = accumulate_covariance_estimators(sc, data, model)

    # Format as ndarrays
    V = len(model.centers)
    A = dict_to_ndarray(A, V)
    mu = dict_to_ndarray(mu, V)
    count = dict_to_ndarray(count, V)

    # Compute params
    R, mu = compute_rotations_from_accumulators(A, mu, count, num_buckets)

    return R, mu, count


def train_subquantizers(sc, split_vecs, M, subquantizer_clusters, model, seed=None):
    """
    Project each data point into it's local space and compute subquantizers by clustering
    each fine split of the locally projected data.
    """
    b = sc.broadcast(model)

    def project_local(x):
        x = np.concatenate(x)
        coarse = b.value.predict_coarse(x)
        return b.value.project(x, coarse)

    projected = split_vecs.map(project_local)

    # Split the vectors into the subvectors
    split_vecs = projected.map(lambda x: np.split(x, M))
    split_vecs.cache()

    subquantizers = []
    for split in xrange(M):
        data = split_vecs.map(lambda x: x[split])
        data.cache()
        sub = KMeans.train(data, subquantizer_clusters, initializationMode='random', maxIterations=10, seed=seed)
        data.unpersist()
        subquantizers.append(np.vstack(sub.clusterCenters))

    return (subquantizers[:len(subquantizers) / 2], subquantizers[len(subquantizers) / 2:])


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


def save_hdfs_proto(m, proto_path):
    """
    Given an LOPQModel object and a path on hdfs, save the model parameters as a protobuf file locally and
    copy the file to the hdfs path.
    """
    print 'Saving protobuf to temp file...'
    f = NamedTemporaryFile(delete=False)
    m.export_proto(f)
    f.close()

    print 'Copying proto file to hdfs...'
    copy_to_hdfs(f, proto_path)
    os.remove(f.name)


def copy_to_hdfs(f, hdfs_path):
    subprocess.call(['hadoop', 'fs', '-copyFromLocal', f.name, hdfs_path])


def validate_arguments(args, model):
    """
    Check provided command line arguments to ensure they are coherent. Provide feedback for potential errors.
    """

    # Parse steps
    args.steps = set(map(int, args.steps.split(',')))

    # Check that the steps make sense
    if STEP_ROTATION not in args.steps and len(args.steps) == 2:
        print 'Training steps invalid'
        sys.exit(1)

    # Find parameters and warn of possibly unintentional discrepancies
    if args.V is None:
        if model is not None:
            args.V = model.V
            print 'Parameter V not specified: using V=%d from provided model.' % model.V
        else:
            print 'Parameter V not specified and no existing model provided. Exiting.'
            sys.exit(1)
    else:
        if model is not None and model.V != args.V:
            if STEP_COARSE in args.steps:
                print 'Parameter V differs between command line argument and provided model: ' + \
                      'coarse quantizers will be trained with V=%d' % args.V
            else:
                print 'Parameter V differs between command line argument and provided model: ' + \
                      'coarse quantizers must be retrained or this discrepancy corrected. Exiting.'
                sys.exit(1)

    if STEP_ROTATION in args.steps or STEP_SUBQUANT in args.steps:
        if args.M is None:
            if model is not None:
                args.M = model.M
                print 'Parameter M not specified: using M=%d from provided model.' % model.M
            else:
                print 'Parameter M not specified and no existing model provided. Exiting.'
                sys.exit(1)
        else:
            if model is not None and model.M != args.M:
                if STEP_ROTATION in args.steps:
                    print 'Parameter M differs between command line argument and provided model: ' + \
                          'model will be trained with M=%d' % args.M
                else:
                    print 'Parameter M differs between command line argument and provided model: ' + \
                          'rotations must be retrained or this discrepancy corrected. Exiting.'
                    sys.exit(1)

    if STEP_ROTATION in args.steps:
        if STEP_COARSE not in args.steps and (model is None or model.Cs is None):
            print 'Cannot train rotations without coarse quantizers. Either train coarse quantizers or provide an existing model. Exiting.'
            sys.exit(1)

    if STEP_SUBQUANT in args.steps:
        if STEP_COARSE not in args.steps and (model is None or model.Cs is None):
            print 'Cannot train subquantizers without coarse quantizers. Either train coarse quantizers or provide an existing model. Exiting.'
            sys.exit(1)
        if STEP_ROTATION not in args.steps and (model is None or model.Rs is None or model.mus is None):
            print 'Cannot train subquantizers without rotations. Either train rotations or provide an existing model. Exiting.'
            sys.exit(1)

    return args

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()

    # Data handling parameters
    parser.add_argument('--data', dest='data', type=str, required=True, help='hdfs path to input data')
    parser.add_argument('--data_udf', dest='data_udf', type=str, default=None, help='module name from which to load a data loading UDF')
    parser.add_argument('--seed', dest='seed', type=int, default=None, help='optional random seed')
    parser.add_argument('--sampling_ratio', dest='sampling_ratio', type=float, default=1.0, help='proportion of data to sample for training')
    parser.add_argument('--subquantizer_sampling_ratio', dest='subquantizer_sampling_ratio', type=float, default=1.0,
                        help='proportion of data to subsample for subquantizer training')

    # Model parameters
    existing_model_group = parser.add_mutually_exclusive_group()
    existing_model_group.add_argument('--existing_model_pkl', dest='existing_model_pkl', type=str, default=None,
                                      help='a pickled LOPQModel from which to extract existing parameters')
    existing_model_group.add_argument('--existing_model_proto', dest='existing_model_proto', type=str, default=None,
                                      help='a protobuf of existing model parameters')

    # Model hyperparameters
    parser.add_argument('--V', dest='V', type=int, default=None, help='number of coarse clusters')
    parser.add_argument('--M', dest='M', type=int, default=None, help='total number of subquantizers')
    parser.add_argument('--subquantizer_clusters', dest='subquantizer_clusters', type=int, default=256, help='number of subquantizer clusters')

    # Training and output directives
    parser.add_argument('--steps', dest='steps', type=str, default='0,1,2',
                        help='comma-separated list of integers indicating which steps of training to perform')
    parser.add_argument('--model_pkl', dest='model_pkl', type=str, default=None, help='hdfs path to save pickle file of resulting LOPQModel')
    parser.add_argument('--model_proto', dest='model_proto', type=str, default=None, help='hdfs path to save protobuf file of resulting model parameters')

    args = parser.parse_args()

    # Check that some output format was provided
    if args.model_pkl is None and args.model_proto is None:
        parser.error('at least one of --model_pkl and --model_proto is required')

    # Load existing model if provided
    model = None
    if args.existing_model_pkl:
        model = pkl.load(open(args.existing_model_pkl))
    elif args.existing_model_proto:
        model = LOPQModel.load_proto(args.existing_model_proto)

    args = validate_arguments(args, model)

    # Build descriptive app name
    get_step_name = lambda x: {STEP_COARSE: 'coarse', STEP_ROTATION: 'rotations', STEP_SUBQUANT: 'subquantizers'}.get(x, None)
    steps_str = ', '.join(filter(lambda x: x is not None, map(get_step_name, sorted(args.steps))))
    APP_NAME = 'LOPQ{V=%d,M=%d}; training %s' % (args.V, args.M, steps_str)

    sc = SparkContext(appName=APP_NAME)

    # Load UDF module if provided and load training data RDD
    if args.data_udf:
        udf_module = __import__(args.data_udf, fromlist=['udf'])
        load_udf = udf_module.udf
        data = load_data(sc, args, data_load_fn=load_udf)
    else:
        data = load_data(sc, args)

    # Initialize parameters
    Cs = Rs = mus = subs = None

    # Get coarse quantizers
    if STEP_COARSE in args.steps:
        Cs = train_coarse(sc, data, args.V, seed=args.seed)
    else:
        Cs = model.Cs

    # Get rotations
    if STEP_ROTATION in args.steps:
        Rs, mus, counts = train_rotations(sc, data, args.M, Cs)
    else:
        Rs = model.Rs
        mus = model.mus

    # Get subquantizers
    if STEP_SUBQUANT in args.steps:
        model = LOPQModel(V=args.V, M=args.M, subquantizer_clusters=args.subquantizer_clusters, parameters=(Cs, Rs, mus, None))

        if args.subquantizer_sampling_ratio != 1.0:
            data = data.sample(False, args.subquantizer_sampling_ratio, args.seed)

        subs = train_subquantizers(sc, data, args.M, args.subquantizer_clusters, model, seed=args.seed)

    # Final output model
    model = LOPQModel(V=args.V, M=args.M, subquantizer_clusters=args.subquantizer_clusters, parameters=(Cs, Rs, mus, subs))

    if args.model_pkl:
        save_hdfs_pickle(model, args.model_pkl)
    if args.model_proto:
        save_hdfs_proto(model, args.model_proto)

    sc.stop()
