"""
Train and evaluate SESEMI architecture for semi-supervised learning
with self-supervised task of recognizing geometric transformations
defined as 90-degree rotations with horizontal and vertical flips.
"""

# Python package imports
import argparse
import os, pickle
import numpy as np
rng = np.random.RandomState(1)
# SESEMI package imports
from datasets import cifar100
from networks import convnet, wrn
from utils import LRScheduler, DenseEvaluator
from utils import gaussian_noise, compile_sesemi
from utils import load_tinyimages, datagen_tinyimages
from utils import global_contrast_normalize, zca_whitener
# Keras package imports
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator


def parse_args():
    """Parse command line input arguments."""
    parser = argparse.ArgumentParser(description='Train and evaluate SESEMI.')
    parser.add_argument('--network', dest='network', type=str, required=True)
    parser.add_argument('--extra', dest='nb_extra', type=int, required=True)
    parser.add_argument('--gpu', dest='gpu_id', type=str, default='0')
    args = parser.parse_args()
    return args


def open_sesemi():
    args = parse_args()
    network = args.network
    nb_extra = args.nb_extra
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    
    arg2var = {'convnet': convnet, 'wrn': wrn,}
    
    # Load Tiny Images.
    # Code adapted from https://github.com/smlaine2/tempens
    with open('./datasets/tiny-images/tiny_index.pkl', 'rb') as f:
        tinyimg_index = pickle.load(f, encoding='latin1')
    
    if nb_extra == 237203:
        print("Using all classes common with CIFAR-100.")
        with open('./datasets/cifar-100/meta', 'rb') as f:
            cifar_labels = pickle.load(f, encoding='latin1')['fine_label_names']
        cifar_to_tinyimg = {'maple_tree': 'maple', 'aquarium_fish': 'fish'}
        cifar_labels = [l if l not in cifar_to_tinyimg else cifar_to_tinyimg[l]
                        for l in cifar_labels]
        load_indices = sum([list(range(*tinyimg_index[label]))
                            for label in cifar_labels], [])
    elif nb_extra == 500000:
        print("Using %d random images." % nb_extra)
        nb_tinyimages = max(e for s, e in tinyimg_index.values())
        load_indices = np.arange(nb_tinyimages)
        rng.shuffle(load_indices)
        load_indices = load_indices[:nb_extra]
        load_indices.sort() # sorted for faster seeks.
    else:
        raise ValueError('`--extra` must be integer 237203 or 500000.')

    print("Loading %d auxiliary unlabeled Tiny Images." % len(load_indices))
    z_train = load_tinyimages(load_indices)
    
    # Load CIFAR-100.
    (x_train, y_train), (x_test, y_test) = cifar100.load_data()

    x_test  = global_contrast_normalize(x_test)
    x_train = global_contrast_normalize(x_train)
    z_train = global_contrast_normalize(z_train)
    
    zca_whiten = zca_whitener(np.concatenate([x_train, z_train], axis=0))
    x_test  = zca_whiten(x_test)
    x_train = zca_whiten(x_train)
    z_train = zca_whiten(z_train)

    x_test  = x_test.reshape((len(x_test), 32, 32, 3))
    x_train = x_train.reshape((len(x_train), 32, 32, 3))
    z_train = z_train.reshape((len(z_train), 32, 32, 3))
    
    y_train = to_categorical(y_train)
    
    # Shared training parameters.
    zca = True
    hflip = True
    epochs = 50
    base_lr = 0.05
    batch_size = 8
    nb_classes = 100
    lr_decay_power = 0.5
    super_dropout = 0.2
    in_network_dropout = 0.0
    input_shape = (32, 32, 3)
    max_iter = (len(x_train) // batch_size) * epochs
    
    # Compile the SESEMI model.
    sesemi_model, inference_model = compile_sesemi(
            arg2var[network], input_shape, nb_classes,
            base_lr, in_network_dropout, super_dropout
        )
    print(sesemi_model.summary())

    lr_poly_decay = LRScheduler(base_lr, max_iter, lr_decay_power)
    evaluate = DenseEvaluator(
            inference_model, (x_test, y_test), hflip, oversample=True)

    super_datagen = ImageDataGenerator(
            width_shift_range=[-2, -1, 0, 1, 2],
            height_shift_range=[-2, -1, 0, 1, 2],
            horizontal_flip=hflip,
            preprocessing_function=gaussian_noise,
            fill_mode='reflect',
        )
    self_datagen = ImageDataGenerator(
            width_shift_range=[-2, -1, 0, 1, 2],
            height_shift_range=[-2, -1, 0, 1, 2],
            horizontal_flip=False,
            preprocessing_function=gaussian_noise,
            fill_mode='reflect',
        )

    super_data = super_datagen.flow(
            x_train, y_train, shuffle=True, batch_size=1, seed=None)
    self_data = self_datagen.flow(
            x_train, shuffle=True, batch_size=1, seed=None)
    extra_data = self_datagen.flow(
            z_train, shuffle=True, batch_size=1, seed=None)
    train_data_loader = datagen_tinyimages(
            super_data, self_data, extra_data, batch_size)

    # Fit the SESEMI model on mini-batches with data augmentation.
    print('Run configuration:')
    print('network=%s,' % network, 'ZCA=%s,' % zca, 'nb_epochs=%d,' % epochs, \
          'horizontal_flip=%s,' % hflip, 'nb_extra=%d,' % len(z_train), \
          'batch_size=%d,' % batch_size, 'gpu_id=%s' % args.gpu_id)
    sesemi_model.fit_generator(train_data_loader,
                               epochs=epochs, verbose=1,
                               steps_per_epoch=len(x_train) // batch_size,
                               callbacks=[lr_poly_decay, evaluate],)


if __name__ == '__main__':
    open_sesemi()

