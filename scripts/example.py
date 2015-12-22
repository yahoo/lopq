# Copyright 2015, Yahoo Inc.
# Licensed under the terms of the Apache License, Version 2.0. See the LICENSE file associated with the project for terms.
import sys
import os

# Add the lopq module - not needed if they are available in the python environment
sys.path.append(os.path.abspath('../python'))

import numpy as np
from sklearn.cross_validation import train_test_split

from lopq import LOPQModel, LOPQSearcher
from lopq.eval import compute_all_neighbors, get_recall
from lopq.model import eigenvalue_allocation


def load_oxford_data():
    from lopq.utils import load_xvecs

    data = load_xvecs('../data/oxford/oxford_features.fvecs')
    return data


def pca(data):
    """
    A simple PCA implementation that demonstrates how eigenvalue allocation
    is used to permute dimensions in order to balance the variance across
    subvectors. There are plenty of PCA implementations elsewhere. What is
    important is that the eigenvalues can be used to compute a variance-balancing
    dimension permutation.
    """

    # Compute mean
    count, D = data.shape
    mu = data.sum(axis=0) / float(count)

    # Compute covariance
    summed_covar = reduce(lambda acc, x: acc + np.outer(x, x), data, np.zeros((D, D)))
    A = summed_covar / (count - 1) - np.outer(mu, mu)

    # Compute eigen decomposition
    eigenvalues, P = np.linalg.eigh(A)

    # Compute a permutation of dimensions to balance variance among 2 subvectors
    permuted_inds = eigenvalue_allocation(2, eigenvalues)

    # Build the permutation into the rotation matrix. One can alternately keep
    # these steps separate, rotating and then permuting, if desired.
    P = P[:, permuted_inds]

    return P, mu


def main():
    """
    A brief demo script showing how to train various LOPQ models with brief
    discussion of trade offs.
    """

    # Get the oxford dataset
    data = load_oxford_data()

    # Compute PCA of oxford dataset. See README in data/oxford for details
    # about this dataset.
    P, mu = pca(data)

    # Mean center and rotate the data; includes dimension permutation.
    # It is worthwhile see how this affects recall performance. On this
    # dataset, which is already PCA'd from higher dimensional features,
    # this additional step to variance balance the dimensions typically
    # improves recall@1 by 3-5%. The benefit can be much greater depending
    # on the dataset.
    data = data - mu
    data = np.dot(data, P)

    # Create a train and test split. The test split will become
    # a set of queries for which we will compute the true nearest neighbors.
    train, test = train_test_split(data, test_size=0.2)

    # Compute distance-sorted neighbors in training set for each point in test set.
    # These will be our groundtruth for recall evaluation.
    nns = compute_all_neighbors(test, train)

    # Fit model
    m = LOPQModel(V=16, M=8)
    m.fit(train, n_init=1)

    # Note that we didn't specify a random seed for fitting the model, so different
    # runs will be different. You may also see a warning that some local projections
    # can't be estimated because too few points fall in a cluster. This is ok for the
    # purposes of this demo, but you might want to avoid this by increasing the amount
    # of training data or decreasing the number of clusters (the V hyperparameter).

    # With a model in hand, we can test it's recall. We populate a LOPQSearcher
    # instance with data and get recall stats. By default, we will retrieve 1000
    # ranked results for each query vector for recall evaluation.
    searcher = LOPQSearcher(m)
    searcher.add_data(train)
    recall, _ = get_recall(searcher, test, nns)
    print 'Recall (V=%d, M=%d, subquants=%d): %s' % (m.V, m.M, m.subquantizer_clusters, str(recall))

    # We can experiment with other hyperparameters without discarding all
    # parameters everytime. Here we train a new model that uses the same coarse
    # quantizers but a higher number of subquantizers, i.e. we increase M.
    m2 = LOPQModel(V=16, M=16, parameters=(m.Cs, None, None, None))
    m2.fit(train, n_init=1)

    # Let's evaluate again.
    searcher = LOPQSearcher(m2)
    searcher.add_data(train)
    recall, _ = get_recall(searcher, test, nns)
    print 'Recall (V=%d, M=%d, subquants=%d): %s' % (m2.V, m2.M, m2.subquantizer_clusters, str(recall))

    # The recall is probably higher. We got better recall with a finer quantization
    # at the expense of more data required for index items.

    # We can also hold both coarse quantizers and rotations fixed and see what
    # increasing the number of subquantizer clusters does to performance.
    m3 = LOPQModel(V=16, M=8, subquantizer_clusters=512, parameters=(m.Cs, m.Rs, m.mus, None))
    m3.fit(train, n_init=1)

    searcher = LOPQSearcher(m3)
    searcher.add_data(train)
    recall, _ = get_recall(searcher, test, nns)
    print 'Recall (V=%d, M=%d, subquants=%d): %s' % (m3.V, m3.M, m3.subquantizer_clusters, str(recall))

    # The recall is probably better than the first but worse than the second. We increased recall
    # only a little by increasing the number of model parameters (double the subquatizer centroids),
    # the index storage requirement (another bit for each fine code), and distance computation time
    # (double the subquantizer centroids).


if __name__ == '__main__':
    main()
