### Oxford dataset

This test dataset is included for tests and the short `example.py` tutorial. The `oxford_features.fvecs` files contains 5063 128-dimensional vectors representing each of the images in the Oxford5k dataset (http://www.robots.ox.ac.uk/~vgg/data/oxbuildings/). The data format is that described here: http://corpus-texmex.irisa.fr/.

The features themselves are the rectified fc7 layer outputs of the BVLC Caffe reference model (https://github.com/BVLC/caffe/tree/master/models/bvlc_reference_caffenet) extracted from resized images and then PCA'd to 128 dimensions from their original 4096.
