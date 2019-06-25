# Semi-Supervised Learning with Self-Supervision

This repository contains a Keras implementation of the SESEMI architecture for semi-supervised image classification, as described in the arXiv paper:

Tran, Phi Vu (2019) Semi-Supervised Learning with Self-Supervised Networks.

![schematic](figure1.png?raw=true)

## Requirements
The code is tested on Ubuntu 16.04 with the following components:

### Software

* Anaconda Python 3.6;
* Keras 2.2.4 using TensorFlow GPU 1.12.0 backend;
* CUDA 9.1 with CuDNN 7.1 acceleration.

### Hardware
This reference implementation loads all data into system memory and utilizes GPU for model training and evaluation. The following hardware specifications are highly recommended:

* At least 64GB of system RAM;
* NVIDIA GeForce GTX TITAN X GPU or better.

## Usage
For training and evaluation, execute the following `bash` commands in the same directory where the code resides:

```bash
# Set the PYTHONPATH environment variable.
$ export PYTHONPATH="/path/to/this/repo:$PYTHONPATH"

# Train and evaluate SESEMI.
$ python train_evaluate_sesemi.py --model <model_str> --dataset <dataset_str> --labels <nb_labels> --gpu <gpu_id>

# Train and evaluate SESEMI with extra data from Tiny Images.
$ python train_evaluate_sesemi_tinyimages.py --model <model_str> --extra <nb_extra> --gpu <gpu_id>
```

The required flags are:

* `<model_str>` refers to either `convnet` or `wrn` architecture;
* `<dataset_str>` refers to one of three supported datasets `svhn`, `cifar10`, and `cifar100`;
* `<nb_labels>` is an integer denoting the number of labeled examples;
* `<nb_extra>` denotes the amount of extra unlabeled data to sample from Tiny Images;
* `<gpu_id>` denotes the GPU device ID, defaults to `0` if only one GPU is available.
