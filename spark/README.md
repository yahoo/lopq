# Spark

This is an implementation of LOPQ training for [Apache Spark](https://spark.apache.org/). Spark's in-memory execution model is well-suited to LOPQ training since there are multiple steps of clustering that involve repeated access to the same data. The scripts provided here run with pyspark and use core functionality implemented in the `lopq` python module.

#### A note about Spark environments

The following usage examples assume that you have a well configured Spark environment suited to the available hardware. Additionally, we assume that the python environment available on both the Spark driver and executors contains all the necessary dependencies, namely the modules listed in `python/requirements.txt` as well as the `lopq` module itself. The [Anaconda](https://www.continuum.io/why-anaconda) environment is a good starting point. At the time of writing, it contains all required dependencies by default except the `protobuf` module, which can be easily installed. To distribute the `lopq` module itself, you could either install it into the environment running on your Spark cluster, or submit it with the Spark job. For example, you can zip the module from the `python/` directory (`zip -r lopq.zip lopq/`) and then submit this zip file with the `--py-files` argument. More information about submitting jobs to Spark is available [here](https://spark.apache.org/docs/latest/submitting-applications.html).

## PCA Training

A recommended preprocessing step for training is to PCA and variance balance the raw data vectors to produce the LOPQ data vectors, i.e. the vectors that LOPQ will quantize. The PCA step is important because it axis-aligns the data and optionally reduces the dimensionality, resulting in better quantization. The variance balancing step permutes the dimensions of the PCA'd vectors so that the first half and second half of the data vectors have roughly the same total variance, which makes the LOPQ coarse codes much better at quantizing the data since each half will be equally "important". The benefit of PCA, dimensionality reduction, and variance balancing in terms of retrieval performance of the downstream LOPQ model will vary based on the data, but it has been seen to provide considerable improvements in many contexts.

The `train_pca.py` script is provided to compute PCA parameters on Spark. It will output a pickled dict of PCA parameters - refer to `train_pca.py` for the contents of this dict. See discussion of data handling in the LOPQ Training section below to learn about loading custom data formats.

After the PCA parameters are computed, the PCA matrix must be truncated to the desired final dimension and the two halves must be variance balanced by permuting the PCA matrix. The `pca_preparation.py` script is provided to do these two preparation steps. Afterwards the training data can be transformed before LOPQ training, perhaps via a data UDF (discussed below).

#### Available parameters

| Command line arg              | Default | Description                                                                    | 
| ----------------------------- | ------- | ------------------------------------------------------------------------------ |
| --data                        | None    | hdfs path to input data                                                        |
| --data_udf                    | None    | optional module name contained a `udf` function to load training data          |
| --seed                        | None    | optional random seed                                                           |
| --sampling_ratio              | 1.0     | proportion of data to sample for training                                      |
| --agg_depth                   | 4       | depth of tree aggregation for computing covariance - increase if you have driver memory issues |
| --output                      | None    | hdfs output path                                                               |


## LOPQ Training

The `train_model.py` script can be configured to run full or partial training of LOPQ models on Spark. The script can resume training from an existing model, using some parameters from the existing model. An existing model can be provided to the script as a pickle file. The `--steps` parameters indicates which steps of training to perform; `0` indicates coarse clustering, `1` indicates rotations fittiing, and `2` indicates subquantizer clustering. The default is for all training steps to be performed.

#### Available parameters

| Command line arg              | Default | Description                                                                    | 
| ----------------------------- | ------- | ------------------------------------------------------------------------------ |
| --data                        | None    | hdfs path to input data                                                        |
| --data_udf                    | None    | optional module name contained a `udf` function to load training data          |
| --seed                        | None    | optional random seed                                                           |
| --sampling_ratio              | 1.0     | proportion of data to sample for training                                      |
| --subquantizer_sampling_ratio | 1.0     | proportion of data to subsample for subquantizer training                      |
| --existing_model_pkl          | None    | a pickled LOPQModel from which to extract existing parameters                  |
| --existing_model_proto        | None    | a protobuf of existing parameters                                              |
| --V                           | None    | number of coarse clusters                                                      |
| --M                           | None    | total number of subquantizers                                                  |
| --subquantizer_clusters       | 256     | number of subquantizer clusters                                                |
| --steps                       | 0,1,2   | comma-separated list of integers indicating which steps of training to perform |
| --model_pkl                   | None    | hdfs path to save pickle file of resulting LOPQModel                           |
| --model_proto                 | None    | hdfs path to save protobuf file of resulting model parameters                  |

#### Usage

Here is an example of training a full model from scratch and saving the model parameters as both a pickle file and a protobuf file:

```bash
spark-submit train_model.py \
	--data /hdfs/path/to/data \
	--V 16 \
	--M 8 \
	--model_pkl /hdfs/output/path/model.pkl \
	--model_proto /hdfs/output/path/model.lopq
```

By providing an existing model, the script can use existing parameters and only the training pipeline for the remaining parameters. This is useful when you want to explore different hyperparameters without retraining everything from scratch. Here is an example of using the coarse quantizers in an existing model and training only rotations and subquantizers. Note that the existing model must be provided to Spark via the `--files` argument. The model can also be provided in protobuf format with `--existing_model_proto`.

```bash
spark-submit \
	--files /path/to/name_of_existing_model.pkl \
	train_model.py \
	--data /hdfs/path/to/data \
	--model_pkl /hdfs/output/path/model.pkl \
	--existing_model_pkl name_of_existing_model.pkl \
	--M 8 \
	--steps 1,2
```

#### Data handling

By default, the training script assumes that your training data is in a text file of tab-delimited `(id, data)` pairs, where the data vector is a base64-encoded pickled numpy array. If this is not the format that your data is in, you can provide the training script a UDF to load the data from your format. This UDF has the following signature:

```python
def udf(sc, data_path, sampling_ratio, seed):
	pass
```

where `sc` is the SparkContext instance, `data_path` is the path provided to the `--data` argument, and `sampling_ratio` and `seed` are the values provided to the arguments of the same name. This UDF must return an RDD of numpy arrays representing the training data and must be named `udf`. An example is provided in `example_udf.py`. The UDF is provided to the script by submitting its module via `--py-files` and passing the module name to the script via `--data-udf`, e.g.:

```bash
spark-submit \
	--py-files example_udf.py \
	train_model.py \
	--data /hdfs/path/to/data \
	--data_udf example_udf \
	--V 16 \
	--M 8 \
	--model_proto /hdfs/output/path/model.lopq
```

## Code Computation

The `compute_codes.py` script takes a fully trained model, an input file of features on hdfs, and an output path on hdfs, and computes LOH codes for all points. The model must be distributed with the job using the `--files` option. The script consumes `(id , data)` pairs and produces a text file of tab-delimited `(id, json-formatted LOPQ code)` pairs, e.g.:

```
33	[[15, 13], [0, 165, 1, 72, 178, 147, 170, 69]]
34	[[5, 9], [104, 227, 160, 185, 248, 152, 170, 126]]
35	[[14, 10], [221, 144, 4, 186, 172, 40, 32, 228]]
36	[[3, 5], [76, 216, 141, 161, 247, 2, 34, 219]]
37	[[0, 5], [205, 140, 214, 194, 39, 229, 131, 0]]
38	[[12, 3], [149, 48, 249, 224, 98, 255, 210, 131]]
```

#### Available parameters

| Command line arg              | Default | Description                                                                    | 
| ----------------------------- | ------- | ------------------------------------------------------------------------------ |
| --data                        | None    | hdfs path to input data                                                        |
| --data_udf                    | None    | optional module name contained a `udf` function to load training data          |
| --seed                        | None    | optional random seed                                                           |
| --sampling_ratio              | 1.0     | proportion of data to sample                                                   |
| --output                      | None    | hdfs output path                                                               |
| --model_pkl                   | None    | file name of the model pickle                                                  |
| --model_proto                 | None    | file name of the model protobuf file                                           |

#### Usage

```bash
spark-submit \
	--files /path/to/name_of_existing_model.pkl \
	compute_codes.py \
	--data /hdfs/path/to/data \
	--output /hdfs/output/path \
	--model_pkl name_of_existing_model.pkl
```

#### Data handling

This script also provides a way for the user to load data from other formats via a UDF. It differs from the training script only in that the output of the UDF must be an RDD of `(id, data)` pairs.
