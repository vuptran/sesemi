"""
Train and evaluate SESEMI architecture for semi-supevised learning
with self-supervised task of recognizing geometric transformations
defined as 90-degree rotations with horizontal and vertical flips.
"""

# Python package imports
import os
import argparse
# Keras package imports
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
# SESEMI package imports
from utils import geometric_transform, global_contrast_normalize
from utils import zca_whitener, stratified_sample, gaussian_noise
from utils import LRScheduler, DenseEvaluator, datagen, open_sesemi
from datasets import svhn, cifar10, cifar100
from models import convnet, wrn


def parse_args():
    """Parse command line input arguments."""
    parser = argparse.ArgumentParser(description='Train and evaluate SESEMI.')
    parser.add_argument('--model', dest='model', type=str, required=True)
    parser.add_argument('--dataset', dest='dataset', type=str, required=True)
    parser.add_argument('--labels', dest='nb_labels', type=int, required=True)
    parser.add_argument('--gpu', dest='gpu_id', type=int, default=0)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    model = args.model
    dataset = args.dataset
    nb_labels = args.nb_labels
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    
    arg2var = {'convnet': convnet,
               'wrn': wrn,
               'svhn': svhn,
               'cifar10': cifar10,
               'cifar100': cifar100,}
    
    # Dataset-specific parameters
    hflip = True
    zca = True
    epochs = 50
    if dataset in ['svhn', 'cifar10']:
        if dataset == 'svhn':
            hflip = False
            zca = False
            epochs = 30
        nb_classes = 10
    elif dataset == 'cifar100':
        nb_classes = 100
    else:
        raise ValueError('`dataset` must be "svhn", "cifar10", "cifar100".')

    (x_train, y_train), (x_test, y_test) = arg2var[dataset].load_data()

    x_train = global_contrast_normalize(x_train)
    x_test = global_contrast_normalize(x_test)
    
    if zca:
        zca_whiten = zca_whitener(x_train)
        x_train = zca_whiten(x_train)
        x_test = zca_whiten(x_test)

    x_train = x_train.reshape((len(x_train), 32, 32, 3))
    x_test = x_test.reshape((len(x_test), 32, 32, 3))
    
    labels_per_class = nb_labels // nb_classes
    if nb_labels == 73257:
        labels_per_class = 1000000
    sample_inds = stratified_sample(y_train, labels_per_class)
    x_labeled = x_train[sample_inds]
    y_labeled = y_train[sample_inds]
    y_labeled = to_categorical(y_labeled)
    
    # Training parameters
    input_shape = (32, 32, 3)
    batch_size = 32
    base_lr = 0.05
    lr_decay_power = 0.5
    dropout_rate = 0.2
    max_iter = (len(x_train) // batch_size) * epochs

    sesemi_model, inference_model = open_sesemi(
        arg2var[model], input_shape, nb_classes, base_lr, dropout_rate)
    print(sesemi_model.summary())

    super_datagen = ImageDataGenerator(
            width_shift_range=3,
            height_shift_range=3,
            horizontal_flip=hflip,
            preprocessing_function=gaussian_noise,
            fill_mode='reflect',
        )
    self_datagen = ImageDataGenerator(
            width_shift_range=3,
            height_shift_range=3,
            horizontal_flip=False,
            preprocessing_function=gaussian_noise,
            fill_mode='reflect',
        )

    super_data = super_datagen.flow(
            x_labeled, y_labeled, shuffle=True, batch_size=1, seed=None)
    self_data = self_datagen.flow(
            x_train, shuffle=True, batch_size=1, seed=None)
    train_data_loader = datagen(super_data, self_data, batch_size)

    lr_poly_decay = LRScheduler(base_lr, max_iter, lr_decay_power)
    evaluate = DenseEvaluator(inference_model, (x_test, y_test), hflip)
    
    # Fit the SESEMI model on mini-batches with data augmentation
    print('Run configuration:')
    print('model=%s,' % model, 'dataset=%s,' % dataset, \
          'horizontal_flip=%s,' % hflip, 'ZCA=%s,' % zca, \
          'nb_epochs=%d,' % epochs, 'batch_size=%d,' % batch_size, \
          'nb_labels=%d,' % len(x_labeled), 'gpu_id=%d' % args.gpu_id)
    sesemi_model.fit_generator(train_data_loader,
                               epochs=epochs, verbose=1,
                               steps_per_epoch=len(x_train) // batch_size,
                               callbacks=[lr_poly_decay, evaluate],)
    return


if __name__ == '__main__':
    main()

