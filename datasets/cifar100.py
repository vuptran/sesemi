"""
Download source CIFAR-100 image classification dataset from:
https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz
"""

import numpy as np
import os
import pickle

        
def load_data(label_mode='fine'):
    """Loads CIFAR-100 dataset.
    
    # Returns
        Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
    """
    if label_mode not in ['fine', 'coarse']:
        raise ValueError('`label_mode` must be "fine" or "coarse".')

    def load_batch(fpath):
        """Internal utility for parsing CIFAR-100 data.
        
        # Returns
        A tuple `(data, labels)`, with `data` normalized between [0, 1]
        """
        with open(fpath, 'rb') as f:
            d = pickle.load(f, encoding='latin1')
        label_key = label_mode + '_labels'
        
        data = d['data']
        labels = d[label_key]
        
        data = data.reshape(len(data), 3, 32, 32).astype('float32') / 255.
        return data, labels
    
    dirname = './datasets/cifar-100'
    
    fpath = os.path.join(dirname, 'train')
    x_train, y_train = load_batch(fpath)

    fpath = os.path.join(dirname, 'test')
    x_test, y_test = load_batch(fpath)
    
    y_train = np.reshape(y_train, (len(y_train),))
    y_test = np.reshape(y_test, (len(y_test),))
    
    x_train = x_train.transpose(0, 2, 3, 1)
    x_test = x_test.transpose(0, 2, 3, 1)
        
    return (x_train, y_train), (x_test, y_test)

