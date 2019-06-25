"""
Download source SVHN cropped digit classification dataset from:
http://ufldl.stanford.edu/housenumbers/
"""

import numpy as np
import scipy.io
import os


def load_data():
    """Loads SVHN dataset.
    
    # Returns
        Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
    """
    dirname = './datasets/svhn'
    
    fpath = os.path.join(dirname, 'train_32x32.mat')
    d = scipy.io.loadmat(fpath)
    x_train = np.transpose(d['X'], (3, 0, 1, 2))
    y_train = d['y'].reshape(-1)
    y_train[y_train == 10] = 0 # Assign label 0 to zero digits

    fpath = os.path.join(dirname, 'test_32x32.mat')
    d = scipy.io.loadmat(fpath)
    x_test = np.transpose(d['X'], (3, 0, 1, 2))
    y_test = d['y'].reshape(-1)
    y_test[y_test == 10] = 0 # Assign label 0 to zero digits

    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    
    return (x_train, y_train), (x_test, y_test)

