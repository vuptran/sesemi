# Learning with Self-Supervised Regularization

### Update: this repo exists only for the reproduction of the experiments presented in the paper below. The user is encouraged to check the PyTorch version of this repo for practical semi-supervised image classification on large realistic datasets using modern CNN backbones, along with the latest developments and features. https://github.com/FlyreelAI/sesemi

This repository contains a Keras implementation of the SESEMI architecture for supervised and semi-supervised image classification, as described in the NeurIPS'19 LIRE Workshop paper:

Tran, Phi Vu (2019) [Exploring Self-Supervised Regularization for Supervised and Semi-Supervised Learning](https://arxiv.org/abs/1906.10343).

## Approach
![schematic](figure1.png?raw=true)

The training and evaluation of the SESEMI architecture for supervised and semi-supervised learning is summarized as follows:

1. Separate the input data into labeled and unlabeled branches. The unlabeled branch consists of all available training examples, but without ground truth label information;
2. Perform geometric transformations on unlabeled data to produce six proxy labels defined as image rotations belonging in the set of {0,90,180,270} degrees along with horizontal (left-right) and vertical (up-down) flips;
3. Apply input data augmentation and noise to each branch independently;
4. At each training step, sample two mini-batches having the same number of unlabeled and labeled examples as inputs to a shared CNN *backbone*. Note that labeled examples will repeat in a mini-batch because the number of unlabeled examples is much greater;
5. Compute the supervised cross-entropy loss using ground truth labels and the self-supervised cross-entropy loss using proxy labels generated from image rotations and flips;
6. Update CNN parameters via stochastic gradient descent by minimizing the sum of supervised and self-supervised loss components;
7. At inference time, take the supervised branch of the network to make predictions on test data and discard the self-supervised branch.

## Requirements
The code is tested on Ubuntu 16.04 with the following components:

### Software

* Anaconda Python 3.6;
* Keras 2.2.4 with TensorFlow GPU 1.12.0 backend;
* CUDA 9.1 with CuDNN 7.1 acceleration.

### Hardware
This reference implementation loads all data into system memory and utilizes GPU for model training and evaluation. The following hardware specifications are highly recommended:

* At least 64GB of system RAM;
* NVIDIA GeForce GTX TITAN X GPU or better.

## Usage
For training and evaluation, execute the following `bash` commands in the same directory where the code resides. Ensure the datasets have been downloaded into their respective [directories](https://github.com/vuptran/sesemi/tree/master/datasets).

```bash
# Set the PYTHONPATH environment variable.
$ export PYTHONPATH="/path/to/this/repo:$PYTHONPATH"

# Train and evaluate SESEMI.
$ python train_evaluate_sesemi.py
	--network <network_str>
	--dataset <dataset_str>
	--labels <nb_labels>
	--gpu <gpu_id>

# Train and evaluate SESEMI with unlabeled extra data from Tiny Images.
$ python train_evaluate_sesemi_tinyimages.py
	--network <network_str>
	--extra <nb_extra>
	--gpu <gpu_id>
```

The required flags are:

* `<network_str>` refers to one of `convnet`, `wrn`, or `nin` architecture;
* `<dataset_str>` refers to one of three supported datasets `svhn`, `cifar10`, and `cifar100`;
* `<nb_labels>` is an integer denoting the number of labeled examples;
* `<nb_extra>` denotes the amount of unlabeled extra data to sample from Tiny Images;
* `<gpu_id>` is a string denoting the GPU device ID, defaults to `0` if not specified.
