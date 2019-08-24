"""Utility functions for semi-supervised learning."""

# Python package imports
import numpy as np
# Set seed number for reproducible data splits.
rng = np.random.RandomState(1)
import os, scipy
from scipy import ndimage
from sklearn.metrics import accuracy_score
# Keras package imports
from keras import optimizers
from keras import initializers
from keras import backend as K
from keras.models import Model
from keras.regularizers import l2
from keras.callbacks import Callback
from keras.utils import to_categorical
from keras.layers import Dropout, Dense, Input
from keras.layers import GlobalAveragePooling2D

proxy_labels = 6


def geometric_transform(image):
    image = np.reshape(image, (32, 32, 3))
    labels = np.empty((proxy_labels,), dtype='uint8')
    images = np.empty((proxy_labels, 32, 32, 3), dtype='float32')
    for i in range(proxy_labels):
        if i <= 3:
            t = np.rot90(image, i)
        elif i == 4:
            t = np.fliplr(image)
        else:
            t = np.flipud(image)
        images[i] = t
        labels[i] = i
    return (images, to_categorical(labels))
        
        
def global_contrast_normalize(images, scale=55, eps=1e-10):
    images = images.astype('float32')
    n, h, w, c = images.shape
    # Flatten images to shape=(nb_images, nb_features)
    images = images.reshape((n, h*w*c))
    # Subtract out the mean of each image
    images -= images.mean(axis=1, keepdims=True)
    # Divide out the norm of each image
    per_image_norm = np.linalg.norm(images, axis=1, keepdims=True)
    # Avoid divide-by-zero
    per_image_norm[per_image_norm < eps] = 1.0
    return float(scale) * images / per_image_norm


def zca_whitener(images, identity_scale=0.1, eps=1e-10):
    """Args:
        images: array of flattened images, shape=(n_images, n_features)
        identity_scale: scalar multiplier for identity in SVD
        eps: small constant to avoid divide-by-zero
    Returns:
        A function which applies ZCA to an array of flattened images
    """
    image_covariance = np.cov(images, rowvar=False)
    U, S, _ = np.linalg.svd(
        image_covariance + identity_scale * np.eye(*image_covariance.shape)
    )
    zca_decomp = np.dot(U, np.dot(np.diag(1.0 / np.sqrt(S + eps)), U.T))
    image_mean = images.mean(axis=0)
    return lambda x: np.dot(x - image_mean, zca_decomp)


def stratified_sample(label_array, labels_per_class):
    label_array = label_array.reshape(-1)
    samples = []
    for cls in range(len(set(label_array))):
        inds = np.where(label_array == cls)[0]
        rng.shuffle(inds)
        inds = inds[:labels_per_class].tolist()
        samples.extend(inds)
    return samples


def gaussian_noise(image, stddev=0.15):
    return image + stddev * np.random.standard_normal(image.shape)


def transform_matrix_offset_center(matrix, x, y):
    o_x = float(x) / 2 + 0.5
    o_y = float(y) / 2 + 0.5
    offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
    reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
    return transform_matrix


def jitter(image, row_axis=0, col_axis=1, channel_axis=2,
           fill_mode='reflect', cval=0.0, order=1):
    tx = np.random.choice([-2, -1, 1, 2])
    ty = np.random.choice([-2, -1, 1, 2])
    
    transform_matrix = np.array([[1, 0, tx],
                                 [0, 1, ty],
                                 [0, 0, 1]])
    h, w = image.shape[row_axis], image.shape[col_axis]
    transform_matrix = transform_matrix_offset_center(
            transform_matrix, h, w)
    image = np.rollaxis(image, channel_axis, 0)
    final_affine_matrix = transform_matrix[:2, :2]
    final_offset = transform_matrix[:2, 2]

    channel_images = [ndimage.interpolation.affine_transform(
        image_channel,
        final_affine_matrix,
        final_offset,
        order=order,
        mode=fill_mode,
        cval=cval) for image_channel in image]
    image = np.stack(channel_images, axis=0)
    image = np.rollaxis(image, 0, channel_axis + 1)
    return image


def datagen(super_iter, self_iter, batch_size):
    """Utility function to load data into required Keras model format."""
    super_batch = batch_size * proxy_labels
    self_batch = batch_size
    while(True):
        x_super, y_super = zip(*[next(super_iter) for _ in range(super_batch)])
        x_self, y_self = zip(*[geometric_transform(next(self_iter))
                               for _ in range(self_batch)])
        x_super = np.vstack(x_super)
        y_super = np.vstack(y_super)
        x_self = np.vstack(x_self)
        y_self = np.vstack(y_self)
        yield ([x_self, x_super], [y_self, y_super])


def datagen_tinyimages(super_iter, self_iter, extra_iter, batch_size):
    """Function to load extra tiny images into required Keras model format."""
    total_batch = 16
    super_batch = total_batch * proxy_labels
    self_batch  = batch_size
    extra_batch = total_batch - self_batch
    while(True):
        x_super, y_super = zip(*[next(super_iter) for _ in range(super_batch)])
        x_self, y_self = zip(*[geometric_transform(next(self_iter))
                               for _ in range(self_batch)])
        x_extra, y_extra = zip(*[geometric_transform(next(extra_iter))
                                 for _ in range(extra_batch)])
        x_super = np.vstack(x_super)
        y_super = np.vstack(y_super)
        x_self = np.vstack(x_self + x_extra)
        y_self = np.vstack(y_self + y_extra)
        yield ([x_self, x_super], [y_self, y_super])


def load_tinyimages(indices):
    dirname = './datasets/tiny-images'
    fpath = os.path.join(dirname, 'tiny_images.bin')
    images = np.empty((len(indices), 3, 32, 32), dtype='float32')
    with open(fpath, 'rb') as f:
        for i, idx in enumerate(indices):
            f.seek(3072 * idx)
            image = np.fromfile(f, dtype='uint8', count=3072)
            images[i] = np.reshape(image, (3, 32, 32))
    images = np.transpose(images, (0, 3, 2, 1)) / 255.
    return images


def compile_sesemi(network, input_shape, nb_classes,
                   lrate, in_network_dropout, super_dropout):
    weight_decay = 0.0005
    initer = initializers.glorot_uniform()

    fc_params = dict(
            use_bias=True,
            activation='softmax',
            kernel_initializer=initer,
            kernel_regularizer=l2(weight_decay),
        )

    cnn_trunk = network.create_network(input_shape, in_network_dropout)
    
    super_in = Input(shape=input_shape, name='super_data')
    self_in  = Input(shape=input_shape, name='self_data')
    super_out = cnn_trunk(super_in)
    self_out  = cnn_trunk(self_in)
    
    super_out = GlobalAveragePooling2D(name='super_gap')(super_out)
    self_out  = GlobalAveragePooling2D(name='self_gap')(self_out)
    if super_dropout > 0.0:
        super_out = Dropout(super_dropout, name='super_dropout')(super_out)
    
    super_out = Dense(nb_classes, name='super_clf', **fc_params)(super_out)
    self_out  = Dense(proxy_labels, name='self_clf', **fc_params)(self_out)

    sesemi_model = Model(inputs=[self_in, super_in],
                         outputs=[self_out, super_out])
    inference_model = Model(inputs=[super_in], outputs=[super_out])

    sgd = optimizers.SGD(lr=lrate, momentum=0.9, nesterov=True)
    sesemi_model.compile(optimizer=sgd,
                         loss={'super_clf': 'categorical_crossentropy',
                               'self_clf' : 'categorical_crossentropy'},
                         loss_weights={'super_clf': 1.0, 'self_clf': 1.0},
                         metrics=None)
    return sesemi_model, inference_model


class LRScheduler(Callback):
    def __init__(self, base_lr, max_iter, power=0.5):
        self.base_lr = base_lr
        self.max_iter = float(max_iter)
        self.power = power
        self.batches = 0
        
    def on_batch_begin(self, batch, logs={}):
        lr = self.base_lr * (1.0 - (self.batches / self.max_iter)) ** self.power
        K.set_value(self.model.optimizer.lr, lr)
        self.batches += 1
        
    def on_epoch_begin(self, epoch, logs={}):
        print('Learning rate: ', K.get_value(self.model.optimizer.lr))


class DenseEvaluator(Callback):
    def __init__(self, inference_model, val_data, hflip,
                 test_every=1, oversample=True):
        x_val = val_data[0]
        y_val = val_data[1]

        self.hflip = hflip
        self.labels = y_val
        self.test_every = test_every
        self.oversample = oversample
        self.inference_model = inference_model

        if not self.oversample:
            self.data = x_val
            return
        
        self.data = []
        for x in x_val:
            t = jitter(x)
            noisy_x = gaussian_noise(x)
            noisy_t = gaussian_noise(t)
            if self.hflip:
                flipx = np.fliplr(x)
                flipt = np.fliplr(t)
                noisy_flipx = gaussian_noise(flipx)
                noisy_flipt = gaussian_noise(flipt)
                self.data.append([x, t, noisy_x, noisy_t,
                                  flipx, flipt, noisy_flipx, noisy_flipt])
            else:
                self.data.append([x, t, noisy_x, noisy_t])
        self.data = np.vstack(self.data)
        
    def on_epoch_end(self, epoch, logs={}):
        if (epoch + 1) % self.test_every != 0:
            return
        
        print('Evaluating with oversample=%s...' % self.oversample)
        y_pred = self.inference_model.predict(self.data, batch_size=128)
        
        if not self.oversample:
            y_pred = np.argmax(y_pred, axis=1)
        else:
            if self.hflip:
                y_pred = y_pred.reshape((len(y_pred) // 8, 8, -1))
            else:
                y_pred = y_pred.reshape((len(y_pred) // 4, 4, -1))
            y_pred = y_pred.mean(axis=1)
            y_pred = np.argmax(y_pred, axis=1)
        
        y_true = self.labels
        
        error = 1.0 - accuracy_score(y_true, y_pred)
        print('classification error rate: {:.4f}'.format(error), '\n')

