# Copyright 2015, Yahoo Inc.
# Licensed under the terms of the Apache License, Version 2.0. See the LICENSE file associated with the project for terms.
from nose.tools import assert_true, assert_equal

import pickle as pkl
import sys
import os
import numpy as np
from sklearn.cross_validation import train_test_split

sys.path.insert(1, os.path.abspath('..'))
from lopq.model import LOPQModel, eigenvalue_allocation, accumulate_covariance_estimators, compute_rotations_from_accumulators
from lopq.search import LOPQSearcher, LOPQSearcherLMDB
from lopq.eval import compute_all_neighbors, get_cell_histogram, get_recall

########################################
# Helpers
########################################


relpath = lambda x: os.path.abspath(os.path.join(os.path.dirname(__file__), x))


def load_oxford_data():
    from lopq.utils import load_xvecs

    data = load_xvecs(relpath('../../data/oxford/oxford_features.fvecs'))
    return data


def make_random_model():
    m = LOPQModel(V=5, M=4, subquantizer_clusters=10)
    m.fit(np.random.RandomState(42).rand(200, 8), n_init=1)
    return m

########################################
# Tests
########################################


def test_eigenvalue_allocation():
    a = pkl.load(open(relpath('./testdata/test_eigenvalue_allocation_input.pkl')))

    vals, vecs = np.linalg.eigh(a)
    res = eigenvalue_allocation(4, vals)

    expected = np.array([
        63, 56, 52, 48, 44, 40, 36, 30, 26, 22, 18, 14, 10, 6, 3, 0,
        62, 57, 53, 51, 45, 41, 39, 33, 32, 31, 29, 25, 21, 17, 13, 9,
        61, 58, 54, 49, 47, 42, 38, 34, 28, 24, 20, 16, 12, 8, 5, 2,
        60, 59, 55, 50, 46, 43, 37, 35, 27, 23, 19, 15, 11, 7, 4, 1
    ])

    assert_true(np.equal(res, expected).all())


def test_eigenvalue_allocation_normalized_features():
    eigenvalues = np.array([
        2.02255824, 1.01940991, 0.01569471, 0.01355569, 0.01264379,
        0.01137654, 0.01108961, 0.01054673, 0.01023358, 0.00989679,
        0.00939045, 0.00900322, 0.00878857, 0.00870027, 0.00850136,
        0.00825236, 0.00813437, 0.00800231, 0.00790201, 0.00782219,
        0.00763405, 0.00752334, 0.00739174, 0.00728246, 0.00701366,
        0.00697365, 0.00677283, 0.00669658, 0.00654397, 0.00647679,
        0.00630645, 0.00621057
    ])
    indices = eigenvalue_allocation(2, eigenvalues)

    first_half = eigenvalues[indices[:16]]
    second_half = eigenvalues[indices[16:]]
    diff = np.abs(np.sum(np.log(first_half)) - np.sum(np.log(second_half)))
    assert_true(diff < .1, "eigenvalue_allocation is not working correctly")


def test_accumulate_covariance_estimators():
    data, centroids = pkl.load(open(relpath('./testdata/test_accumulate_covariance_estimators_input.pkl')))
    expected = pkl.load(open(relpath('./testdata/test_accumulate_covariance_estimators_output.pkl')))

    actual = accumulate_covariance_estimators(data, centroids)

    # Summed residual outer products
    assert_true(np.allclose(expected[0], actual[0]))

    # Summed residuals
    assert_true(np.allclose(expected[1], actual[1]))

    # Assignment count per cluster
    assert_true(np.array_equal(expected[2], actual[2]))

    # Assignments over data
    assert_true(np.array_equal(expected[3], actual[3]))

    # Residual data
    assert_true(np.allclose(expected[4], actual[4]))


def test_compute_rotations_from_accumulators():

    A, mu, count, num_buckets = pkl.load(open(relpath('./testdata/test_compute_rotations_from_accumulators_input.pkl')))
    expected = pkl.load(open(relpath('./testdata/test_compute_rotations_from_accumulators_output.pkl')))

    actual = compute_rotations_from_accumulators(A, mu, count, num_buckets)

    # Rotations
    assert_true(np.allclose(expected[0], actual[0]))

    # Mean residuals
    assert_true(np.allclose(expected[1], actual[1]))


def test_reconstruction():
    m = LOPQModel.load_proto(relpath('./testdata/random_test_model.lopq'))

    code = ((0, 1), (0, 1, 2, 3))
    r = m.reconstruct(code)
    expected = [-2.27444688, 6.47126941, 4.5042611, 4.76683476, 0.83671082, 9.36027283, 8.11780532, 6.34846377]

    assert_true(np.allclose(expected, r))


def test_oxford5k():

    random_state = 40
    data = load_oxford_data()
    train, test = train_test_split(data, test_size=0.2, random_state=random_state)

    # Compute distance-sorted neighbors in training set for each point in test set
    nns = compute_all_neighbors(test, train)

    # Fit model
    m = LOPQModel(V=16, M=8)
    m.fit(train, n_init=1, random_state=random_state)

    # Assert correct code computation
    assert_equal(m.predict(test[0]), ((3, 2), (14, 164, 83, 49, 185, 29, 196, 250)))

    # Assert low number of empty cells
    h = get_cell_histogram(train, m)
    assert_equal(np.count_nonzero(h == 0), 6)

    # Assert true NN recall on test set
    searcher = LOPQSearcher(m)
    searcher.add_data(train)
    recall, _ = get_recall(searcher, test, nns)
    assert_true(np.all(recall > [0.51, 0.92, 0.97, 0.97]))

    # Test partial fitting with just coarse quantizers
    m2 = LOPQModel(V=16, M=8, parameters=(m.Cs, None, None, None))
    m2.fit(train, n_init=1, random_state=random_state)

    searcher = LOPQSearcher(m2)
    searcher.add_data(train)
    recall, _ = get_recall(searcher, test, nns)
    assert_true(np.all(recall > [0.51, 0.92, 0.97, 0.97]))

    # Test partial fitting with coarse quantizers and rotations
    m3 = LOPQModel(V=16, M=8, parameters=(m.Cs, m.Rs, m.mus, None))
    m3.fit(train, n_init=1, random_state=random_state)

    searcher = LOPQSearcher(m3)
    searcher.add_data(train)
    recall, _ = get_recall(searcher, test, nns)
    assert_true(np.all(recall > [0.51, 0.92, 0.97, 0.97]))


def test_proto():
    import os

    filename = './temp_proto.lopq'
    m = make_random_model()
    m.export_proto(filename)
    m2 = LOPQModel.load_proto(filename)

    assert_equal(m.V, m2.V)
    assert_equal(m.M, m2.M)
    assert_equal(m.subquantizer_clusters, m2.subquantizer_clusters)

    assert_true(np.allclose(m.Cs[0], m2.Cs[0]))
    assert_true(np.allclose(m.Rs[0], m2.Rs[0]))
    assert_true(np.allclose(m.mus[0], m2.mus[0]))
    assert_true(np.allclose(m.subquantizers[0][0], m.subquantizers[0][0]))

    os.remove(filename)


def test_mat():
    import os

    filename = './temp_mat.mat'
    m = make_random_model()
    m.export_mat(filename)
    m2 = LOPQModel.load_mat(filename)

    assert_equal(m.V, m2.V)
    assert_equal(m.M, m2.M)
    assert_equal(m.subquantizer_clusters, m2.subquantizer_clusters)

    assert_true(np.allclose(m.Cs[0], m2.Cs[0]))
    assert_true(np.allclose(m.Rs[0], m2.Rs[0]))
    assert_true(np.allclose(m.mus[0], m2.mus[0]))
    assert_true(np.allclose(m.subquantizers[0][0], m.subquantizers[0][0]))

    os.remove(filename)


def searcher_instance_battery(searcher, q):
    """
    Helper to perform battery of searcher tests.
    """
    retrieved, visited = searcher.get_result_quota(q)
    assert_equal(len(retrieved), 12)
    assert_equal(visited, 3)

    retrieved, visited = searcher.search(q)
    assert_equal(len(retrieved), 10)
    assert_equal(visited, 3)

    retrieved, visited = searcher.get_result_quota(q, quota=20)
    assert_equal(len(retrieved), 28)
    assert_equal(visited, 5)

    retrieved, visited = searcher.search(q, quota=20)
    assert_equal(len(retrieved), 20)
    assert_equal(visited, 5)

    retrieved, visited = searcher.search(q, quota=20, limit=10)
    assert_equal(len(retrieved), 10)
    assert_equal(visited, 5)


def test_searcher():
    data = pkl.load(open(relpath('./testdata/test_searcher_data.pkl')))
    m = LOPQModel.load_proto(relpath('./testdata/random_test_model.lopq'))

    q = np.ones(8)

    # Test add_data
    searcher = LOPQSearcher(m)
    searcher.add_data(data)
    searcher_instance_battery(searcher, q)

    # Test add_codes
    searcher = LOPQSearcher(m)
    codes = [m.predict(x) for x in data]
    searcher.add_codes(codes)
    searcher_instance_battery(searcher, q)


def test_searcher_lmdb():
    import shutil

    data = pkl.load(open(relpath('./testdata/test_searcher_data.pkl')))
    m = LOPQModel.load_proto(relpath('./testdata/random_test_model.lopq'))

    lmbd_test_path = './test_lopq_lmbd'
    q = np.ones(8)

    # Test add_data
    searcher = LOPQSearcherLMDB(m, lmbd_test_path)
    searcher.add_data(data)
    searcher_instance_battery(searcher, q)

    # Clean up
    shutil.rmtree(lmbd_test_path)

    # Test add_codes
    searcher = LOPQSearcherLMDB(m, lmbd_test_path)
    codes = [m.predict(x) for x in data]
    searcher.add_codes(codes)
    searcher_instance_battery(searcher, q)

    # Clean up
    shutil.rmtree(lmbd_test_path)

