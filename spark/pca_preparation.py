import cPickle as pkl
import base64
import numpy as np
from lopq.model import eigenvalue_allocation


def main(args):

    params = pkl.load(open(args.pca_params))
    P = params['P']
    E = params['E']
    mu = params['mu']

    # Reduce dimension
    E = E[-args.D:]
    P = P[:,-args.D:]

    # Balance variance across halves
    permuted_inds = eigenvalue_allocation(2, E)
    P = P[:, permuted_inds]

    # Save new params
    pkl.dump({ 'P': P, 'mu': mu }, open(args.output, 'w'))


def apply_PCA(x, mu, P):
    """
    Example of applying PCA.
    """
    return np.dot(x - mu, P)


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()

    parser.add_argument('--pca_params', dest='pca_params', type=str, required=True, help='path to pickle file of PCA parameters')
    parser.add_argument('--D', dest='D', type=int, default=128, help='desired final feature dimension')
    parser.add_argument('--output', dest='output', type=str, required=True, help='path to pickle file of new PCA parameters')
    args = parser.parse_args()

    main(args)
