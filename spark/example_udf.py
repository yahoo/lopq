# Copyright 2015, Yahoo Inc.
# Licensed under the terms of the Apache License, Version 2.0. See the LICENSE file associated with the project for terms.
import cPickle as pkl
import base64


def udf(sc, data_path, sampling_ratio, seed):
    """
    This is an example UDF function to load training data. It loads data from a text file
    with base64-encoded pickled numpy arrays on each line.
    """

    # Compute the number of cores in our cluster - used below to heuristically set the number of partitions
    total_cores = int(sc._conf.get('spark.executor.instances')) * int(sc._conf.get('spark.executor.cores'))

    # Load and sample down the dataset
    d = sc.textFile(data_path, total_cores * 3).sample(False, sampling_ratio, seed)

    deserialize_vec = lambda s: pkl.loads(base64.decodestring(s))
    vecs = d.map(deserialize_vec)

    return vecs
