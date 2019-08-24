"""
Train and evaluate SESEMI architecture for semi-supevised learning
with self-supervised task of recognizing geometric transformations
defined as 90-degree rotations with horizontal and vertical flips.
"""

# Python package imports
import os, argparse
# SESEMI package imports
from networks import convnet, wrn, nin
from datasets import svhn, cifar10, cifar100
from utils import global_contrast_normalize, zca_whitener
from utils import stratified_sample, gaussian_noise, datagen
from utils import LRScheduler, DenseEvaluator, compile_sesemi
# Keras package imports
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator


def parse_args():
    """Parse command line input arguments."""
    parser = argparse.ArgumentParser(description='Train and evaluate SESEMI.')
    parser.add_argument('--network', dest='network', type=str, required=True)
    parser.add_argument('--dataset', dest='dataset', type=str, required=True)
    parser.add_argument('--labels', dest='nb_labels', type=int, required=True)
    parser.add_argument('--gpu', dest='gpu_id', type=str, default='0')
    args = parser.parse_args()
    return args


def open_sesemi():
    args = parse_args()
    network = args.network
    dataset = args.dataset
    nb_labels = args.nb_labels
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    
    arg2var = {'convnet': convnet,
               'wrn': wrn,
               'nin': nin,
               'svhn': svhn,
               'cifar10': cifar10,
               'cifar100': cifar100,}
    
    # Experiment- and dataset-dependent parameters.
    zca = True
    hflip = True
    epochs = 50
    if dataset in {'svhn', 'cifar10'}:
        if dataset == 'svhn':
            zca = False
            hflip = False
            epochs = 30
        nb_classes = 10
    elif dataset == 'cifar100':
        nb_classes = 100
    else:
        raise ValueError('`dataset` must be "svhn", "cifar10", "cifar100".')
    super_dropout = 0.2
    in_network_dropout = 0.0
    if network == 'convnet' and dataset == 'svhn':
        super_dropout = 0.5
        in_network_dropout = 0.5
    elif network == 'wrn' and dataset == 'svhn':
        super_dropout = 0.5

    # Prepare the dataset.
    (x_train, y_train), (x_test, y_test) = arg2var[dataset].load_data()
    
    x_test  = global_contrast_normalize(x_test)
    x_train = global_contrast_normalize(x_train)
    
    if zca:
        zca_whiten = zca_whitener(x_train)
        x_train = zca_whiten(x_train)
        x_test  = zca_whiten(x_test)

    x_test  = x_test.reshape((len(x_test), 32, 32, 3))
    x_train = x_train.reshape((len(x_train), 32, 32, 3))
    
    if nb_labels in {50000, 73257}:
        x_labeled = x_train
        y_labeled = y_train
    else:
        labels_per_class = nb_labels // nb_classes
        sample_inds = stratified_sample(y_train, labels_per_class)
        x_labeled = x_train[sample_inds]
        y_labeled = y_train[sample_inds]
    
    y_labeled = to_categorical(y_labeled)

    # Shared training parameters.
    base_lr = 0.05
    batch_size = 16
    lr_decay_power = 0.5
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
            x_labeled, y_labeled, shuffle=True, batch_size=1, seed=None)
    self_data = self_datagen.flow(
            x_train, shuffle=True, batch_size=1, seed=None)
    train_data_loader = datagen(super_data, self_data, batch_size)
    
    # Fit the SESEMI model on mini-batches with data augmentation.
    print('Run configuration:')
    print('network=%s,' % network, 'dataset=%s,' % dataset, \
          'horizontal_flip=%s,' % hflip, 'ZCA=%s,' % zca, \
          'nb_epochs=%d,' % epochs, 'batch_size=%d,' % batch_size, \
          'nb_labels=%d,' % len(y_labeled), 'gpu_id=%s' % args.gpu_id)
    sesemi_model.fit_generator(train_data_loader,
                               epochs=epochs, verbose=1,
                               steps_per_epoch=len(x_train) // batch_size,
                               callbacks=[lr_poly_decay, evaluate],)


if __name__ == '__main__':
    open_sesemi()

