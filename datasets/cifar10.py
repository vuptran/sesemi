"""
Download CIFAR-10 image classification dataset from:
https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
"""

import numpy as np
import os, pickle


def load_data():
    """Loads CIFAR-10 dataset.
    
    # Returns
        Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
    """
    def load_batch(fpath):
        """Internal utility for parsing CIFAR-10 data.
        
        # Returns
        A tuple `(data, labels)`, with `data` normalized between [0, 1].
        """
        with open(fpath, 'rb') as f:
            d = pickle.load(f, encoding='latin1')
        data = d['data']
        labels = d['labels']
        
        data = data.reshape(len(data), 3, 32, 32).astype('float32') / 255.
        return data, labels
    
    dirname = './datasets/cifar-10'
    nb_train_samples = 50000
    
    x_train = np.empty((nb_train_samples, 3, 32, 32), dtype='float32')
    y_train = np.empty((nb_train_samples,), dtype='uint8')

    for i in range(1, 6):
        fpath = os.path.join(dirname, 'data_batch_' + str(i))
        (x_train[(i - 1) * 10000: i * 10000, :, :, :],
         y_train[(i - 1) * 10000: i * 10000]) = load_batch(fpath)
    
    fpath = os.path.join(dirname, 'test_batch')
    x_test, y_test = load_batch(fpath)
    
    y_test = np.reshape(y_test, (len(y_test),))

    x_train = x_train.transpose(0, 2, 3, 1)
    x_test = x_test.transpose(0, 2, 3, 1)
        
    return (x_train, y_train), (x_test, y_test)

