# Python LOPQ module

This module implements training and testing of LOPQ models along with a variety of other utilities useful for evaluation and data management. It includes a simple implementation of approximate nearest neighbor search with an LOPQ index.

## Installation

```python
pip install lopq
```

## Usage

```python
from lopq import LOPQModel, LOPQSearcher

# Define a model and fit it to data
model = LOPQModel(V=8, M=4)
model.fit(data)

# Compute the LOPQ codes for a vector
code = model.predict(x)

# Create a searcher to index data with the model
searcher = LOPQSearcher(model)
searcher.add_data(data)

# Retrieve ranked nearest neighbors
nns = searcher.search(x, quota=100)
```

A more detailed usage walk-through is found in `scripts/example.py`.

## Training

Refer to the documentation in the `model` submodules and, in particular, the `LOPQModel` class for more usage information.

Available parameters for fitting data:

| Name                      | Default | Description                                                               |
| ------------------------- | ------- | ------------------------------------------------------------------------- |
| V                         | 8       | The number of clusters per coarse quantizer.                              |
| M                         | 4       | The total number of fine codes.                                           |
| kmeans_coarse_iters       | 10      | The number of iterations of k-means for training coarse quantizers.       |
| kmeans_local_iters        | 20      | The number of iterations of k-means for training subquantizers.           |
| subquantizer_clusters     | 256     | The number of clusters to train per subquantizer.                         |
| subquantizer_sample_ratio | 1.0     | The ratio of the data to sample for training subquantizers.               |
| random_state              | None    | A seed for seeding random operations during training.                     |
| parameters                | None    | A tuple of trained model parameters to instantiate the model with.        |
| verbose                   | False   | A boolean indicating whether to produce verbose output.                   |

## Submodules

There are a handful of submodules, here is a brief description of each.

| Submodule      | Description |
| -------------- | ----------- |
| model          | Core training algorithm and the `LOPQModel` class that encapsulates model parameters.
| search         | An implementation of the multisequence algorithm for retrieval on a multi-index as well as the `LOPQSearcher` class, a simple Python implementation of an LOPQ search index and LOPQ ranking. |
| eval           | Functions to aid in evaluating and benchmarking trained LOPQ models. |
| utils          | Miscellaneous utility functions. |
| lopq_model_pb2 | Protobuf generated module. |
