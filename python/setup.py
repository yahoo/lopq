#!/usr/bin/env python

# Copyright 2015, Yahoo Inc.
# Licensed under the terms of the Apache License, Version 2.0. See the LICENSE file associated with the project for terms.
import os
import json
from setuptools import setup


# Package Metadata filename
METADATA_FILENAME = 'lopq/package_metadata.json'
BASEPATH = os.path.dirname(os.path.abspath(__file__))


# Long description of package
LONG_DESCRIPTION = """
# Locally Optimized Product Quantization

This is Python training and testing code for Locally Optimized Product Quantization (LOPQ) models, as well as Spark scripts to scale training to hundreds of millions of vectors. The resulting model can be used in Python with code provided here or deployed via a Protobuf format to, e.g., search backends for high performance approximate nearest neighbor search.

### Overview

Locally Optimized Product Quantization (LOPQ) [1] is a hierarchical quantization algorithm that produces codes of configurable length for data points. These codes are efficient representations of the original vector and can be used in a variety of ways depending on application, including as hashes that preserve locality, as a compressed vector from which an approximate vector in the data space can be reconstructed, and as a representation from which to compute an approximation of the Euclidean distance between points.

Conceptually, the LOPQ quantization process can be broken into 4 phases. The training process also fits these phases to the data in the same order.

1. The raw data vector is PCA'd to `D` dimensions (possibly the original dimensionality). This allows subsequent quantization to more efficiently represent the variation present in the data.
2. The PCA'd data is then product quantized [2] by two k-means quantizers. This means that each vector is split into two subvectors each of dimension `D / 2`, and each of the two subspaces is quantized independently with a vocabulary of size `V`. Since the two quantizations occur independently, the dimensions of the vectors are permuted such that the total variance in each of the two subspaces is approximately equal, which allows the two vocabularies to be equally important in terms of capturing the total variance of the data. This results in a pair of cluster ids that we refer to as "coarse codes".
3. The residuals of the data after coarse quantization are computed. The residuals are then locally projected independently for each coarse cluster. This projection is another application of PCA and dimension permutation on the residuals and it is "local" in the sense that there is a different projection for each cluster in each of the two coarse vocabularies. These local rotations make the next and final step, another application of product quantization, very efficient in capturing the variance of the residuals.
4. The locally projected data is then product quantized a final time by `M` subquantizers, resulting in `M` "fine codes". Usually the vocabulary for each of these subquantizers will be a power of 2 for effective storage in a search index. With vocabularies of size 256, the fine codes for each indexed vector will require `M` bytes to store in the index.

The final LOPQ code for a vector is a `(coarse codes, fine codes)` pair, e.g. `((3, 2), (14, 164, 83, 49, 185, 29, 196, 250))`.

### Nearest Neighbor Search

A nearest neighbor index can be built from these LOPQ codes by indexing each document into its corresponding coarse code bucket. That is, each pair of coarse codes (which we refer to as a "cell") will index a bucket of the vectors quantizing to that cell.

At query time, an incoming query vector undergoes substantially the same process. First, the query is split into coarse subvectors and the distance to each coarse centroid is computed. These distances can be used to efficiently compute a priority-ordered sequence of cells [3] such that cells later in the sequence are less likely to have near neighbors of the query than earlier cells. The items in cell buckets are retrieved in this order until some desired quota has been met.

After this retrieval phase, the fine codes are used to rank by approximate Euclidean distance. The query is projected into each local space and the distance to each indexed item is estimated as the sum of the squared distances of the query subvectors to the corresponding subquantizer centroid indexed by the fine codes.

NN search with LOPQ is highly scalable and has excellent properties in terms of both index storage requirements and query-time latencies when implemented well.

#### References

For more information and performance benchmarks can be found at http://image.ntua.gr/iva/research/lopq/.

1. Y. Kalantidis, Y. Avrithis. [Locally Optimized Product Quantization for Approximate Nearest Neighbor Search.](http://image.ntua.gr/iva/files/lopq.pdf) CVPR 2014.
2. H. Jegou, M. Douze, and C. Schmid. [Product quantization for nearest neighbor search.](https://lear.inrialpes.fr/pubs/2011/JDS11/jegou_searching_with_quantization.pdf) PAMI, 33(1), 2011.
3. A. Babenko and V. Lempitsky. [The inverted multi-index.](http://www.computer.org/csdl/trans/tp/preprint/06915715.pdf) CVPR 2012.
"""

# Create a dictionary of our arguments, this way this script can be imported
# without running setup() to allow external scripts to see the setup settings.
setup_arguments = {
    'name': 'lopq',
    'version': '1.0.0',
    'author': 'Clayton Mellina,Yannis Kalantidis,Huy Nguyen',
    'author_email': 'clayton@yahoo-inc.com',
    'url': 'http://github.com/yahoo/lopq',
    'license': 'Apache-2.0',
    'keywords': ['lopq', 'locally optimized product quantization', 'product quantization', 'compression', 'ann', 'approximate nearest neighbor', 'similarity search'],
    'packages': ['lopq'],
    'long_description': LONG_DESCRIPTION,
    'description': 'Python code for training and deploying Locally Optimized Product Quantization (LOPQ) for approximate nearest neighbor search of high dimensional data.',
    'classifiers': [
            'Development Status :: 5 - Production/Stable',
            'License :: OSI Approved :: Apache Software License',
            'Intended Audience :: Developers',
            'Intended Audience :: Science/Research',
            'Natural Language :: English',
            'Operating System :: MacOS :: MacOS X',
            'Operating System :: Microsoft :: Windows',
            'Operating System :: POSIX',
            'Operating System :: POSIX',
            'Operating System :: Unix',
            'Programming Language :: Python :: 2.7',
            'Topic :: Scientific/Engineering',
            'Topic :: Software Development'
    ],
    'package_data': {
        'lopq': ['package_metadata.json']
    },
    'platforms': 'Windows,Linux,Solaris,Mac OS-X,Unix',
    'include_package_data': True,
    'install_requires': ['protobuf>=2.6', 'numpy>=1.9', 'scipy>=0.14', 'scikit-learn>=0.15', 'lmdb>=0.87']
}


class Git(object):
    """
    Simple wrapper class to the git command line tools
    """
    version_list = ['0', '7', '0']

    def __init__(self, version=None):
        if version:
            self.version_list = version.split('.')

    @property
    def version(self):
        """
        Generate a Unique version value from the git information
        :return:
        """
        git_rev = len(os.popen('git rev-list HEAD').readlines())
        if git_rev != 0:
            self.version_list[-1] = '%d' % git_rev
        version = '.'.join(self.version_list)
        return version

    @property
    def branch(self):
        """
        Get the current git branch
        :return:
        """
        return os.popen('git rev-parse --abbrev-ref HEAD').read().strip()

    @property
    def hash(self):
        """
        Return the git hash for the current build
        :return:
        """
        return os.popen('git rev-parse HEAD').read().strip()

    @property
    def origin(self):
        """
        Return the fetch url for the git origin
        :return:
        """
        for item in os.popen('git remote -v'):
            split_item = item.strip().split()
            if split_item[0] == 'origin' and split_item[-1] == '(push)':
                return split_item[1]


def add_scripts_to_package():
    """
    Update the "scripts" parameter of the setup_arguments with any scripts
    found in the "scripts" directory.
    :return:
    """
    global setup_arguments

    if os.path.isdir('scripts'):
        setup_arguments['scripts'] = [
            os.path.join('scripts', f) for f in os.listdir('scripts')
        ]


def get_and_update_package_metadata():
    """
    Update the package metadata for this package if we are building the package.
    :return:metadata - Dictionary of metadata information
    """
    global setup_arguments
    global METADATA_FILENAME

    if not os.path.exists('.git') and os.path.exists(METADATA_FILENAME):
        with open(METADATA_FILENAME) as fh:
            metadata = json.load(fh)
    else:
        git = Git(version=setup_arguments['version'])
        metadata = {
            'version': git.version,
            'git_hash': git.hash,
            'git_origin': git.origin,
            'git_branch': git.branch
        }
        with open(METADATA_FILENAME, 'w') as fh:
            json.dump(metadata, fh)
    return metadata


if __name__ == '__main__':
    # We're being run from the command line so call setup with our arguments
    metadata = get_and_update_package_metadata()
    setup_arguments['version'] = metadata['version']
    setup(**setup_arguments)
