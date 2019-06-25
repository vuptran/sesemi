"""13-Layer Max-Pooling ConvNet with BatchNormalization."""

from keras.models import Model
from keras.layers import Input, Conv2D, Dropout
from keras.layers import MaxPooling2D, LeakyReLU
from keras.layers import BatchNormalization
from keras.regularizers import l2
from keras import initializers
from keras import backend as K

seed_number = 1
# Non-linearity params
leakiness = 0.0
# Batchnorm params
mom = 0.99
eps = 0.001
gamma = 'ones'
# Convolution params
bias = True
weight_decay = 0.0005
initer = initializers.he_normal(seed=seed_number)


def create_model(input_shape, dropout=0.0):
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    
    data = Input(shape=input_shape) 

    x = Conv2D(128, (3, 3), padding='same', kernel_initializer=initer,
               kernel_regularizer=l2(weight_decay), use_bias=bias)(data)
    x = BatchNormalization(axis=channel_axis, momentum=mom,
                           epsilon=eps, gamma_initializer=gamma)(x)
    x = LeakyReLU(leakiness)(x)

    x = Conv2D(128, (3, 3), padding='same', kernel_initializer=initer,
               kernel_regularizer=l2(weight_decay), use_bias=bias)(x)
    x = BatchNormalization(axis=channel_axis, momentum=mom,
                           epsilon=eps, gamma_initializer=gamma)(x)
    x = LeakyReLU(leakiness)(x)

    x = Conv2D(128, (3, 3), padding='same', kernel_initializer=initer,
               kernel_regularizer=l2(weight_decay), use_bias=bias)(x)
    x = BatchNormalization(axis=channel_axis, momentum=mom,
                           epsilon=eps, gamma_initializer=gamma)(x)
    x = LeakyReLU(leakiness)(x)

    x = MaxPooling2D((2, 2), padding='same')(x)
    if dropout > 0.0: x = Dropout(dropout)(x)

    x = Conv2D(256, (3, 3), padding='same', kernel_initializer=initer,
               kernel_regularizer=l2(weight_decay), use_bias=bias)(x)
    x = BatchNormalization(axis=channel_axis, momentum=mom,
                           epsilon=eps, gamma_initializer=gamma)(x)
    x = LeakyReLU(leakiness)(x)

    x = Conv2D(256, (3, 3), padding='same', kernel_initializer=initer,
               kernel_regularizer=l2(weight_decay), use_bias=bias)(x)
    x = BatchNormalization(axis=channel_axis, momentum=mom,
                           epsilon=eps, gamma_initializer=gamma)(x)
    x = LeakyReLU(leakiness)(x)

    x = Conv2D(256, (3, 3), padding='same', kernel_initializer=initer,
               kernel_regularizer=l2(weight_decay), use_bias=bias)(x)
    x = BatchNormalization(axis=channel_axis, momentum=mom,
                           epsilon=eps, gamma_initializer=gamma)(x)
    x = LeakyReLU(leakiness)(x)

    x = MaxPooling2D((2, 2), padding='same')(x)
    if dropout > 0.0: x = Dropout(dropout)(x)

    x = Conv2D(512, (3, 3), padding='valid', kernel_initializer=initer,
               kernel_regularizer=l2(weight_decay), use_bias=bias)(x)
    x = BatchNormalization(axis=channel_axis, momentum=mom,
                           epsilon=eps, gamma_initializer=gamma)(x)
    x = LeakyReLU(leakiness)(x)

    x = Conv2D(256, (1, 1), padding='valid', kernel_initializer=initer,
               kernel_regularizer=l2(weight_decay), use_bias=bias)(x)
    x = BatchNormalization(axis=channel_axis, momentum=mom,
                           epsilon=eps, gamma_initializer=gamma)(x)
    x = LeakyReLU(leakiness)(x)

    x = Conv2D(128, (1, 1), padding='valid', kernel_initializer=initer,
               kernel_regularizer=l2(weight_decay), use_bias=bias)(x)
    x = BatchNormalization(axis=channel_axis, momentum=mom,
                           epsilon=eps, gamma_initializer=gamma)(x)
    x = LeakyReLU(leakiness)(x)

    # Return output dimensions 6 x 6 x 128
    model = Model(data, x, name='convnet_trunk')
    return model
    
