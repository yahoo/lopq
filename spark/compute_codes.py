# Copyright 2015, Yahoo Inc.
# Licensed under the terms of the Apache License, Version 2.0. See the LICENSE file associated with the project for terms.
from pyspark.context import SparkContext

import cPickle as pkl
import base64
import json

from lopq.model import LOPQModel


def default_data_loading(sc, data_path, sampling_ratio, seed):
    """
    This function loads data from a text file, sampling it by the provided
    ratio and random seed, and interprets each line as a tab-separated (id, data) pair
    where 'data' is assumed to be a base64-encoded pickled numpy array.
    The data is returned as an RDD of (id, numpy array) tuples.
    """
    # Compute the number of cores in our cluster - used below to heuristically set the number of partitions
    total_cores = int(sc._conf.get('spark.executor.instances')) * int(sc._conf.get('spark.executor.cores'))

    # Load and sample down the dataset
    d = sc.textFile(data_path, total_cores * 3).sample(False, sampling_ratio, seed)

    # The data is (id, vector) tab-delimited pairs where each vector is
    # a base64-encoded pickled numpy array
    d = d.map(lambda x: x.split('\t')).map(lambda x: (x[0], pkl.loads(base64.decodestring(x[1]))))

    return d


def main(sc, args, data_load_fn=default_data_loading):

    # Load model
    model = None
    if args.model_pkl:
        model = pkl.load(open(args.model_pkl))
    elif args.model_proto:
        model = LOPQModel.load_proto(args.model_proto)

    # Load data
    d = data_load_fn(sc, args.data, args.sampling_ratio, args.seed)

    # Distribute model instance
    m = sc.broadcast(model)

    # Compute codes and convert to string
    codes = d.map(lambda x: (x[0], m.value.predict(x[1]))).map(lambda x: '%s\t%s' % (x[0], json.dumps(x[1])))

    codes.saveAsTextFile(args.output)

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()

    # Data handling parameters
    parser.add_argument('--data', dest='data', type=str, default=None, required=True, help='hdfs path to input data')
    parser.add_argument('--data_udf', dest='data_udf', type=str, default=None, help='module name from which to load a data loading UDF')
    parser.add_argument('--seed', dest='seed', type=int, default=None, help='optional random seed for sampling')
    parser.add_argument('--sampling_ratio', dest='sampling_ratio', type=float, default=1.0, help='proportion of data to sample for model application')
    parser.add_argument('--output', dest='output', type=str, default=None, required=True, help='hdfs path to output data')

    existing_model_group = parser.add_mutually_exclusive_group(required=True)
    existing_model_group.add_argument('--model_pkl', dest='model_pkl', type=str, default=None, help='a pickled LOPQModel to evaluate on the data')
    existing_model_group.add_argument('--model_proto', dest='model_proto', type=str, default=None, help='a protobuf LOPQModel to evaluate on the data')

    args = parser.parse_args()

    sc = SparkContext(appName='LOPQ code computation')

    # Load UDF module if provided
    if args.data_udf:
        udf_module = __import__(args.data_udf, fromlist=['udf'])
        load_udf = udf_module.udf
        main(sc, args, data_load_fn=load_udf)
    else:
        main(sc, args)

    sc.stop()
