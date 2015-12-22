# Copyright 2015, Yahoo Inc.
# Licensed under the terms of the Apache License, Version 2.0. See the LICENSE file associated with the project for terms.
import numpy as np
from sklearn.cluster import KMeans
import logging
import sys
from collections import namedtuple
from .utils import iterate_splits, predict_cluster

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)
logger.addHandler(logging.StreamHandler(sys.stdout))


########################################
# Core training algo
########################################

def eigenvalue_allocation(num_buckets, eigenvalues):
    """
    Compute a permutation of eigenvalues to balance variance accross buckets
    of dimensions.

    Described in section 3.2.4 in http://research.microsoft.com/pubs/187499/cvpr13opq.pdf

    Note, the following slides indicate this function will break when fed eigenvalues < 1
    without the scaling trick implemented below:

        https://www.robots.ox.ac.uk/~vgg/rg/slides/ge__cvpr2013__optimizedpq.pdf


    :param int num_buckets:
        the number of dimension buckets over which to allocate eigenvalues
    :param ndarray eigenvalues:
        a vector of eigenvalues

    :returns ndarray:
        a vector of indices by which to permute the eigenvectors
    """
    D = len(eigenvalues)
    dims_per_bucket = D / num_buckets
    eigenvalue_product = np.zeros(num_buckets, dtype=float)
    bucket_size = np.zeros(num_buckets, dtype=int)
    permutation = np.zeros((num_buckets, dims_per_bucket), dtype=int)

    # We first must scale the eigenvalues by dividing by their
    # smallets non-zero value to avoid problems with the algorithm
    # when eigenvalues are less than 1.
    min_non_zero_eigenvalue = np.min(np.abs(eigenvalues[np.nonzero(eigenvalues)]))
    eigenvalues = eigenvalues / min_non_zero_eigenvalue

    # Iterate eigenvalues in descending order
    sorted_inds = np.argsort(eigenvalues)[::-1]
    log_eigs = np.log2(abs(eigenvalues))
    for ind in sorted_inds:

        # Find eligible (not full) buckets
        eligible = (bucket_size < dims_per_bucket).nonzero()

        # Find eligible bucket with least eigenvalue product
        i = eigenvalue_product[eligible].argmin(0)
        bucket = eligible[0][i]

        # Update eigenvalue product for this bucket
        eigenvalue_product[bucket] = eigenvalue_product[bucket] + log_eigs[ind]

        # Store bucket assignment and update size
        permutation[bucket, bucket_size[bucket]] = ind
        bucket_size[bucket] += 1

    return np.reshape(permutation, D)


def compute_local_rotations(data, C, num_buckets):
    """
    Compute a rotation matrix for each cluster in the model by estimating
    the covariance matrix for each cluster and balancing variance across
    buckets of the dimensions.

    :param ndarray data:
        data points, each row an observation
    :param ndarray C:
        a VxD matrix of cluster centroids
    :param int num_buckets:
        the number of buckets accross which to balance variance in the rotation matrix

    :returns ndarray:
        a VxDxD tensor containing the rotation matrix for each cluster
        where V is the number of clusters and D the dimension of the data (which is split here)
    :returns ndarray:
        a Nx1 vector of cluster assignments for each data point
    :returns ndarray:
        a VxD vector of mean residuals for each cluster
    :returns ndarray:
        an NxD matrix of residuals
    """

    logger.info('Fitting local rotations...')

    A, mu, count, assignments, residuals = accumulate_covariance_estimators(data, C)

    R, mu = compute_rotations_from_accumulators(A, mu, count, num_buckets)

    logger.info('Done fitting local rotations.')

    return R, mu, assignments, residuals


def accumulate_covariance_estimators(data, C):
    """
    Accumulate covariance estimators for each cluster with a pass through the data.

    :param ndarray data:
        NxD array - observations on the rows
    :param ndarray C:
        VxD array of cluster centroids

    :returns ndarray A:
        VxDxD array - total sum of residual outer products for each cluster
    :returns ndarray mu:
        VxD array of total sum of residuals per cluster
    :returns ndarray count:
        Vx1 array of cluster sizes
    :returns ndarray assignments:
        Nx1 array of cluster assignments
    :returns ndarray residuals:
        NxD array of data residuals
    """

    V = C.shape[0]
    N = data.shape[0]
    D = data.shape[1]

    # Essential variables
    A = np.zeros((V, D, D))                 # accumulators for covariance estimator per cluster
    mu = np.zeros((V, D))                   # residual means
    count = np.zeros(V, dtype=int)          # count of points per cluster
    assignments = np.zeros(N, dtype=int)    # point cluster assignments
    residuals = np.zeros((N, D))            # residual for data points given cluster assignment

    # Iterate data points, accumulate estimators
    for i in xrange(N):
        d = data[i]

        # Find cluster assignment and residual
        cluster = predict_cluster(d, C)
        centroid = C[cluster]
        residual = d - centroid
        assignments[i] = cluster

        # Accumulate estimators for covariance matrix for the assigned cluster
        mu[cluster] += residual
        count[cluster] += 1
        A[cluster] += np.outer(residual, residual)
        residuals[i] = residual

    return A, mu, count, assignments, residuals


def compute_rotations_from_accumulators(A, mu, count, num_buckets):
    """
    Given accumulators computed on cluster residuals, compute the optimal
    rotation matrix. The A and mu variables are modified in place and returned to
    avoid memory allocation.

    :param ndarray A:
        a VxDxD array - total sum of outer products of residuals per cluster
    :param ndarray mu:
        a VxD array - total sum of residuals per cluster
    :param ndarray count:
        a Vx1 array - count of points for each cluster
    :param int num_buckets:
        the number of subvectors to balance variance across

    :returns ndarray A:
        a VxDxD array - per cluster local rotations of size DxD on the first dimension
    :returns ndarray mu:
        a VxD array of mean residuals per cluster
    """

    V, D = mu.shape

    # For each cluster, use accumulator variables to estimate covariance matrix
    # and compute rotation matrix
    for i in xrange(V):

        # Normalize
        num_points = count[i]
        mu[i] /= num_points

        # Compute covariance estimator
        cov = (A[i] + A[i].transpose()) / (2 * (num_points - 1)) - np.outer(mu[i], mu[i])

        # Compute eigenvalues, reuse A matrix
        if num_points < D:
            logger.warn('Fewer points (%d) than dimensions (%d) in rotation computation for cluster %d' % (num_points, D, i))
            eigenvalues = np.ones(D)
            A[i] = np.eye(D)
        else:
            eigenvalues, A[i] = np.linalg.eigh(cov)

        # Permute eigenvectors to balance variance in subquantizers
        permuted_inds = eigenvalue_allocation(num_buckets, eigenvalues)
        A[i] = A[i, :, permuted_inds]

    return A, mu


def project_residuals_to_local(residuals, assignments, Rs, mu):
    """
    Given residuals for training datapoints, their cluster assignments,
    the mean of cluster residuals, and the cluster rotation matrices, project
    all residuals to their appropriate local frame. This is run to generate
    points to cluster for subquantizer training.

    :params ndarray residuals:
        an NxD array of residuals
    :params ndarray assignments:
        an Nx1 array of cluster ids assignments for residuals
    :params ndarray Rs:
        a VxDxD array of rotation matrices for each cluster
    :params ndarray mu:
        a VxD matrix of mean residuals for each cluster

    :returns ndarray:
        an NxD array of locally projected residuals
    """
    projected = np.zeros(residuals.shape)
    for i in xrange(residuals.shape[0]):
        res = residuals[i]
        a = assignments[i]
        projected[i] = np.dot(Rs[a], res - mu[a])

    return projected


def compute_residuals(data, C):
    assignments = np.apply_along_axis(predict_cluster, 1, data, C)
    residuals = data - C[assignments]
    return residuals, assignments


def train_coarse(data, V=8, kmeans_coarse_iters=10, n_init=10, random_state=None):
    """
    Train a kmeans model.

    :param ndarray data:
        an NxD array with observations on the rows
    :param int V:
        the number of clusters
    :param int kmeans_coarse_iters:
        the nubmer of iterations
    :param int random_state:
        a random state to seed the clustering

    :returns ndarray:
        a VxD matrix of cluster centroids
    """

    logger.info('Fitting coarse quantizer...')

    # Fit coarse model
    model = KMeans(n_clusters=V, init="k-means++", max_iter=kmeans_coarse_iters, n_init=n_init, n_jobs=1, verbose=False, random_state=random_state)
    model.fit(data)

    logger.info('Done fitting coarse quantizer.')

    return model.cluster_centers_


def train_subquantizers(data, num_buckets, subquantizer_clusters=256, kmeans_local_iters=20, n_init=10, random_state=None):
    """
    Fit a set of num_buckets subquantizers for corresponding subvectors.
    """

    subquantizers = list()
    for i, d in enumerate(np.split(data, num_buckets, axis=1)):
        model = KMeans(n_clusters=subquantizer_clusters, init="k-means++", max_iter=kmeans_local_iters,
                       n_init=n_init, n_jobs=1, verbose=False, random_state=random_state)
        model.fit(d)
        subquantizers.append(model.cluster_centers_)
        logger.info('Fit subquantizer %d of %d.' % (i + 1, num_buckets))

    return subquantizers


def train(data, V=8, M=4, subquantizer_clusters=256, parameters=None,
          kmeans_coarse_iters=10, kmeans_local_iters=20, n_init=10,
          subquantizer_sample_ratio=1.0, random_state=None, verbose=False):
    """
    Fit an LOPQ model.

    :param ndarray data:
        a NxD matrix of training data points with observations on the rows

    :param int V:
        number of coarse clusters
    :param int M:
        number of fine codes; same as number of bytes per compressed
        vector in memory with 256 subquantizer clusters
    :param int subquantizer_clusters:
        the number of clusters for each subquantizer
    :param tuple parameters:
        a tuple of parameters - missing parameters are allowed to be None

    :param int kmeans_coarse_iters:
        kmeans iterations
    :param int kmeans_local_iters:
        kmeans iterations for subquantizers
    :param int n_init:
        the number of independent kmeans runs for all kmeans when training - set low for faster training
    :param float subquantizer_sample_ratio:
        the proportion of the training data to sample for training subquantizers - since the number of
        subquantizer clusters is much smaller then the number of coarse clusters, less data is needed
    :param int random_state:
        a random seed used in all random operations during training if provided
    :param bool verbose:
        a bool enabling verbose output during training

    :returns tuple:
        a tuple of model parameters that can be used to instantiate an LOPQModel object
    """

    # Set logging level for verbose mode
    if (verbose):
        logger.setLevel(logging.DEBUG)

    # Extract parameters
    Cs = Rs = mus = subquantizers = None
    if parameters is not None:
        Cs, Rs, mus, subquantizers = parameters

    # Enforce parameter dependencies
    if Rs is None or mus is None:
        Rs = mus = None

    # Split vectors
    # TODO: permute dims here if this hasn't already been done
    first_half, second_half = np.split(data, 2, axis=1)

    # Cluster coarse splits
    if Cs is not None:
        logger.info('Using existing coarse quantizers.')
        C1, C2 = Cs
    else:
        C1 = train_coarse(first_half, V, kmeans_coarse_iters, n_init, random_state)
        C2 = train_coarse(second_half, V, kmeans_coarse_iters, n_init, random_state)

    # Compute local rotations
    if Rs is not None and mus is not None:
        logger.info('Using existing rotations.')
        Rs1, Rs2 = Rs
        mu1, mu2 = mus
        assignments1 = assignments2 = residuals1 = residuals2 = None
    else:
        Rs1, mu1, assignments1, residuals1 = compute_local_rotations(first_half, C1, M / 2)
        Rs2, mu2, assignments2, residuals2 = compute_local_rotations(second_half, C2, M / 2)

    # Subquantizers don't need as much data, so we could sample here
    subquantizer_sample_ratio = min(subquantizer_sample_ratio, 1.0)
    N = data.shape[0]
    N2 = int(np.floor(subquantizer_sample_ratio * N))
    sample_inds = np.random.RandomState(random_state).choice(N, N2, False)
    logger.info('Sampled training data for subquantizers with %f proportion (%d points).' % (subquantizer_sample_ratio, N2))

    # Use assignments and residuals from rotation computation if available
    if assignments1 is not None:
        residuals1 = residuals1[sample_inds]
        residuals2 = residuals2[sample_inds]
        assignments1 = assignments1[sample_inds]
        assignments2 = assignments2[sample_inds]
    else:
        residuals1, assignments1 = compute_residuals(first_half[sample_inds], C1)
        residuals2, assignments2 = compute_residuals(second_half[sample_inds], C2)

    # Project residuals
    logger.info('Projecting residuals to local frame...')
    projected1 = project_residuals_to_local(residuals1, assignments1, Rs1, mu1)
    projected2 = project_residuals_to_local(residuals2, assignments2, Rs2, mu2)

    logger.info('Fitting subquantizers...')
    subquantizers1 = train_subquantizers(projected1, M / 2, subquantizer_clusters, kmeans_local_iters, n_init, random_state=random_state)
    subquantizers2 = train_subquantizers(projected2, M / 2, subquantizer_clusters, kmeans_local_iters, n_init, random_state=random_state)
    logger.info('Done fitting subquantizers.')

    return (C1, C2), (Rs1, Rs2), (mu1, mu2), (subquantizers1, subquantizers2)

########################################
# Model class
########################################

# Named tuple type for LOH codes
LOPQCode = namedtuple('LOPQCode', ['coarse', 'fine'])


class LOPQModel(object):
    def __init__(self, V=8, M=4, subquantizer_clusters=256, parameters=None):
        """
        Create an LOPQModel instance that encapsulates a complete LOPQ model with parameters and hyperparameters.

        :param int V:
            the number of clusters per a coarse split
        :param int M:
            the total number of subvectors (equivalent to the total number of subquantizers)
        :param int subquantizer_clusters:
            the number of clusters for each subquantizer
        :param tuple parameters:
            a tuple of parameters - missing parameters are allowed to be None

            the tuple will look like the following

            ((C1, C2), (Rs1, Rs2), (mu1, mu2), (subquantizers1, subquantizers2))

            where each element is itself a pair with one split of parameters for the each of the coarse splits.

            the parameters have the following data types (V and M have the meaning described above,
            D is the total dimension of the data, and S is the number of subquantizer clusters):

                C: VxD/2 ndarray of coarse centroids
                R: VxD/2xD/2 ndarray of fitted rotation matrices for each coarse cluster
                mu: VxD/2 ndarray of mean residuals for each coar cluster
                subquantizer: length M/2 list of SxD/M ndarrays of cluster centroids for each subvector
        """

        # If learned parameters are passed in explicitly, derive the model params by inspection.
        self.Cs, self.Rs, self.mus, self.subquantizers = parameters if parameters is not None else (None, None, None, None)

        if self.Cs is not None:
            self.V = self.Cs[0].shape[0]
            self.num_coarse_splits = len(self.Cs)
        else:
            self.V = V
            self.num_coarse_splits = 2

        if self.subquantizers is not None:
            self.num_fine_splits = len(self.subquantizers[0])
            self.M = self.num_fine_splits * self.num_coarse_splits
            self.subquantizer_clusters = self.subquantizers[0][0].shape[0]
        else:
            self.num_fine_splits = M / 2
            self.M = M
            self.subquantizer_clusters = subquantizer_clusters

    def fit(self, data, kmeans_coarse_iters=10, kmeans_local_iters=20, n_init=10, subquantizer_sample_ratio=1.0, random_state=None, verbose=False):
        """
        Fit a model with the current model parameters. This method will use existing parameters and only
        train missing parameters.

        :param int kmeans_coarse_iters:
            the number of kmeans iterations for coarse quantizer training
        :param int kmeans_local_iters:
            the number of kmeans iterations for subquantizer taining
        :param int n_init:
            the number of independent kmeans runs for all kmeans when training - set low for faster training
        :param float subquantizer_sample_ratio:
            the proportion of the training data to sample for training subquantizers - since the number of
            subquantizer clusters is much smaller then the number of coarse clusters, less data is needed
        :param int random_state:
            a random seed used in all random operations during training if provided
        :param bool verbose:
            a bool enabling verbose output during training
        """
        existing_parameters = (self.Cs, self.Rs, self.mus, self.subquantizers)

        parameters = train(data, self.V, self.M, self.subquantizer_clusters, existing_parameters,
                           kmeans_coarse_iters, kmeans_local_iters, n_init, subquantizer_sample_ratio,
                           random_state, verbose)

        self.Cs, self.Rs, self.mus, self.subquantizers = parameters

    def get_split_parameters(self, split):
        """
        A helper to return parameters for a given coarse split.

        :params int split:
            the coarse split

        :returns ndarray:
            a matrix of centroids for the coarse model
        :returns list:
            a list of residual means for each cluster
        :returns list:
            a list of rotation matrices for each cluster
        :returns list:
            a list of centroid matrices for each subquantizer in this coarse split
        """
        return self.Cs[split] if self.Cs is not None else None, \
            self.Rs[split] if self.Rs is not None else None, \
            self.mus[split] if self.mus is not None else None, \
            self.subquantizers[split] if self.subquantizers is not None else None

    def predict(self, x):
        """
        Compute both coarse and fine codes for a datapoint.

        :param ndarray x:
            the point to code

        :returns tuple:
            a tuple of coarse codes
        :returns tuple:
            a tuple of fine codes
        """
        # Compute coarse quantizer codes
        coarse_codes = self.predict_coarse(x)

        # Compute fine codes
        fine_codes = self.predict_fine(x, coarse_codes)

        return LOPQCode(coarse_codes, fine_codes)

    def predict_coarse(self, x):
        """
        Compute the coarse codes for a datapoint.

        :param ndarray x:
            the point to code

        :returns tuple:
            a tuple of coarse codes
        """
        return tuple([predict_cluster(cx, self.Cs[split]) for cx, split in iterate_splits(x, self.num_coarse_splits)])

    def predict_fine(self, x, coarse_codes=None):
        """
        Compute the fine codes for a datapoint.

        :param ndarray x:
            the point to code
        :param ndarray coarse_codes:
            the coarse codes for the point
            if they are already computed

        :returns tuple:
            a tuple of fine codes
        """
        if coarse_codes is None:
            coarse_codes = self.predict_coarse(x)

        px = self.project(x, coarse_codes)

        fine_codes = []
        for cx, split in iterate_splits(px, self.num_coarse_splits):

            # Get product quantizer parameters for this split
            _, _, _, subC = self.get_split_parameters(split)

            # Compute subquantizer codes
            fine_codes += [predict_cluster(fx, subC[sub_split]) for fx, sub_split in iterate_splits(cx, self.num_fine_splits)]

        return tuple(fine_codes)

    def project(self, x, coarse_codes, coarse_split=None):
        """
        Project this vector to its local residual space defined by the coarse codes.

        :param ndarray x:
            the point to project
        :param ndarray coarse_codes:
            the coarse codes defining the local space
        :param int coarse_split:
            index of the coarse split to get distances for - if None then all splits
            are computed

        :returns ndarray:
            the projected vector
        """
        px = []

        if coarse_split is None:
            split_iter = iterate_splits(x, self.num_coarse_splits)
        else:
            split_iter = [(np.split(x, self.num_coarse_splits)[coarse_split], coarse_split)]

        for cx, split in split_iter:

            # Get product quantizer parameters for this split
            C, R, mu, _ = self.get_split_parameters(split)

            # Retrieve already computed coarse cluster
            cluster = coarse_codes[split]

            # Compute residual
            r = cx - C[cluster]

            # Project residual to local frame
            pr = np.dot(R[cluster], r - mu[cluster])
            px.append(pr)

        return np.concatenate(px)

    def reconstruct(self, codes):
        """
        Given a code tuple, reconstruct an approximate vector.

        :param tuple codes:
            a code tuple as returned from the predict method

        :returns ndarray:
            a reconstructed vector
        """
        coarse_codes, fine_codes = codes

        x = []
        for fc, split in iterate_splits(fine_codes, self.num_coarse_splits):

            # Get product quantizer parameters for this split
            C, R, mu, subC = self.get_split_parameters(split)

            # Concatenate the cluster centroids for this split of fine codes
            sx = reduce(lambda acc, c: np.concatenate((acc, subC[c[0]][c[1]])), enumerate(fc), [])

            # Project residual out of local space
            cluster = coarse_codes[split]
            r = np.dot(R[cluster].transpose(), sx) + mu[cluster]

            # Reconstruct from cluster centroid
            x = np.concatenate((x, r + C[cluster]))

        return x

    def get_subquantizer_distances(self, x, coarse_codes, coarse_split=None):
        """
        Project a given query vector to the local space of the given coarse codes
        and compute the distances of each subvector to the corresponding subquantizer
        clusters.

        :param ndarray x:
            a query  vector
        :param tuple coarse_codes:
            the coarse codes defining which local space to project to
        :param int coarse_split:
            index of the coarse split to get distances for - if None then all splits
            are computed

        :returns list:
            a list of distances to each subquantizer cluster for each subquantizer
        """

        px = self.project(x, coarse_codes)
        subquantizer_dists = []

        if coarse_split is None:
            split_iter = iterate_splits(px, self.num_coarse_splits)
        else:
            split_iter = [(np.split(px, self.num_coarse_splits)[coarse_split], coarse_split)]

        # for cx, split in iterate_splits(px, self.num_coarse_splits):
        for cx, split in split_iter:
            _, _, _, subC = self.get_split_parameters(split)
            subquantizer_dists += [((fx - subC[sub_split]) ** 2).sum(axis=1) for fx, sub_split in iterate_splits(cx, self.num_fine_splits)]

        return subquantizer_dists

    def get_cell_id_for_coarse_codes(self, coarse_codes):
        return coarse_codes[1] + coarse_codes[0] * self.V

    def get_coarse_codes_for_cell_id(self, cell_id):
        return (int(np.floor(float(cell_id) / self.V)), cell_id % self.V)

    def export_mat(self, filename):
        """
        Export model parameters in .mat file format.

        Splits in the parameters (coarse splits and fine splits) are concatenated together in the
        resulting arrays. For example, the Cs paramaters become a 2 x V x D array where the first dimension
        indexes the split. The subquantizer centroids are encoded similarly as a 2 x (M/2) x 256 x (D/M) array.
        """
        from scipy.io import savemat
        from .utils import concat_new_first

        Cs = concat_new_first(self.Cs)
        Rs = concat_new_first(self.Rs)
        mus = concat_new_first(self.mus)
        subs = concat_new_first(map(concat_new_first, self.subquantizers))

        savemat(filename, {'Cs': Cs, 'Rs': Rs, 'mus': mus, 'subs': subs, 'V': self.V, 'M': self.M})

    @staticmethod
    def load_mat(filename):
        """
        Reconstitute an LOPQModel in the format exported by the `export_mat` method above.
        """
        from scipy.io import loadmat

        d = loadmat(filename)

        M = d['M'][0][0]
        Cs = tuple(map(np.squeeze, np.split(d['Cs'], 2, axis=0)))
        Rs = tuple(map(np.squeeze, np.split(d['Rs'], 2, axis=0)))
        mus = tuple(map(np.squeeze, np.split(d['mus'], 2, axis=0)))

        subs = tuple([map(np.squeeze, np.split(half, M / 2, axis=0)) for half in map(np.squeeze, np.split(d['subs'], 2, axis=0))])

        return LOPQModel(parameters=(Cs, Rs, mus, subs))

    def export_proto(self, f):
        """
        Export model parameters in protobuf format.
        """
        from .lopq_model_pb2 import LOPQModelParams
        from itertools import chain

        lopq_params = LOPQModelParams()
        lopq_params.D = 2 * self.Cs[0].shape[1]
        lopq_params.V = self.V
        lopq_params.M = self.M
        lopq_params.num_subquantizers = self.subquantizer_clusters

        def matrix_from_ndarray(m, a):
            m.values.extend(map(float, np.nditer(a, order='C')))
            m.shape.extend(a.shape)
            return m

        def vector_from_ndarray(m, a):
            m.values.extend(map(float, np.nditer(a, order='C')))
            return m

        for C in self.Cs:
            matrix_from_ndarray(lopq_params.Cs.add(), C)
        for R in chain(*self.Rs):
            matrix_from_ndarray(lopq_params.Rs.add(), R)
        for mu in chain(*self.mus):
            vector_from_ndarray(lopq_params.mus.add(), mu)
        for sub in chain(*self.subquantizers):
            matrix_from_ndarray(lopq_params.subs.add(), sub)

        if type(f) is str:
            f = open(f, 'wb')
        f.write(lopq_params.SerializeToString())
        f.close()

    @staticmethod
    def load_proto(filename):
        """
        Reconstitute a model from parameters stored in protobuf format.
        """
        from .lopq_model_pb2 import LOPQModelParams
        from .utils import concat_new_first

        def halves(arr):
            return [arr[:len(arr) / 2], arr[len(arr) / 2:]]

        lopq_params = LOPQModelParams()

        try:
            f = open(filename)
            lopq_params.ParseFromString(f.read())
            f.close()

            Cs = [np.reshape(C.values, C.shape) for C in lopq_params.Cs]
            Rs = map(concat_new_first, halves([np.reshape(R.values, R.shape) for R in lopq_params.Rs]))
            mus = map(concat_new_first, halves([np.array(mu.values) for mu in lopq_params.mus]))
            subs = halves([np.reshape(sub.values, sub.shape) for sub in lopq_params.subs])

            return LOPQModel(parameters=(Cs, Rs, mus, subs))

        except IOError:
            print filename + ": Could not open file."
            return None
