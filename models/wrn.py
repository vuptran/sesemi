"""Wide Residual Networks with Max Pooling.
Reference - https://arxiv.org/abs/1605.07146
"""

from keras.models import Model
from keras.layers import Input, Add, MaxPooling2D
from keras.layers import Conv2D, Dropout, LeakyReLU
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


def initial_conv(input):
    x = Conv2D(16, (3, 3), padding='same', kernel_initializer=initer,
               kernel_regularizer=l2(weight_decay), use_bias=bias)(input)

    channel_axis = 1 if K.image_data_format() == "channels_first" else -1

    x = BatchNormalization(axis=channel_axis, momentum=mom,
                           epsilon=eps, gamma_initializer=gamma)(x)
    x = LeakyReLU(leakiness)(x)
    return x


def expand_conv(init, base, k, strides=(1, 1)):
    x = Conv2D(base * k, (3, 3), padding='same',
               strides=strides, kernel_initializer=initer,
               kernel_regularizer=l2(weight_decay), use_bias=bias)(init)

    channel_axis = 1 if K.image_data_format() == "channels_first" else -1

    x = BatchNormalization(axis=channel_axis, momentum=mom,
                           epsilon=eps, gamma_initializer=gamma)(x)
    x = LeakyReLU(leakiness)(x)

    x = Conv2D(base * k, (3, 3), padding='same', kernel_initializer=initer,
               kernel_regularizer=l2(weight_decay), use_bias=bias)(x)

    skip = Conv2D(base * k, (1, 1), padding='same',
                  strides=strides, kernel_initializer=initer,
                  kernel_regularizer=l2(weight_decay), use_bias=bias)(init)

    m = Add()([x, skip])
    return m


def conv1_block(input, k=1, dropout=0.0):
    init = input

    channel_axis = 1 if K.image_data_format() == "channels_first" else -1

    x = BatchNormalization(axis=channel_axis, momentum=mom,
                           epsilon=eps, gamma_initializer=gamma)(input)
    x = LeakyReLU(leakiness)(x)
    x = Conv2D(16 * k, (3, 3), padding='same', kernel_initializer=initer,
               kernel_regularizer=l2(weight_decay), use_bias=bias)(x)

    if dropout > 0.0: x = Dropout(dropout)(x)

    x = BatchNormalization(axis=channel_axis, momentum=mom,
                           epsilon=eps, gamma_initializer=gamma)(x)
    x = LeakyReLU(leakiness)(x)
    x = Conv2D(16 * k, (3, 3), padding='same', kernel_initializer=initer,
               kernel_regularizer=l2(weight_decay), use_bias=bias)(x)

    m = Add()([init, x])
    return m


def conv2_block(input, k=1, dropout=0.0):
    init = input

    channel_axis = 1 if K.image_dim_ordering() == "th" else -1

    x = BatchNormalization(axis=channel_axis, momentum=mom,
                           epsilon=eps, gamma_initializer=gamma)(input)
    x = LeakyReLU(leakiness)(x)
    x = Conv2D(32 * k, (3, 3), padding='same', kernel_initializer=initer,
               kernel_regularizer=l2(weight_decay), use_bias=bias)(x)

    if dropout > 0.0: x = Dropout(dropout)(x)

    x = BatchNormalization(axis=channel_axis, momentum=mom,
                           epsilon=eps, gamma_initializer=gamma)(x)
    x = LeakyReLU(leakiness)(x)
    x = Conv2D(32 * k, (3, 3), padding='same', kernel_initializer=initer,
               kernel_regularizer=l2(weight_decay), use_bias=bias)(x)

    m = Add()([init, x])
    return m


def conv3_block(input, k=1, dropout=0.0):
    init = input

    channel_axis = 1 if K.image_dim_ordering() == "th" else -1

    x = BatchNormalization(axis=channel_axis, momentum=mom,
                           epsilon=eps, gamma_initializer=gamma)(input)
    x = LeakyReLU(leakiness)(x)
    x = Conv2D(64 * k, (3, 3), padding='same', kernel_initializer=initer,
               kernel_regularizer=l2(weight_decay), use_bias=bias)(x)

    if dropout > 0.0: x = Dropout(dropout)(x)

    x = BatchNormalization(axis=channel_axis, momentum=mom,
                           epsilon=eps, gamma_initializer=gamma)(x)
    x = LeakyReLU(leakiness)(x)
    x = Conv2D(64 * k, (3, 3), padding='same', kernel_initializer=initer,
               kernel_regularizer=l2(weight_decay), use_bias=bias)(x)

    m = Add()([init, x])
    return m


def create_model(input_dim, N=4, k=2, dropout=0.0):
    """
    For WRN depth 16: set N = (16 - 4) / 6 = 2
    For WRN depth 28: set N = (28 - 4) / 6 = 4
    For WRN depth 40: set N = (40 - 4) / 6 = 6
    param k is width of the network.
    """
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1

    ip = Input(shape=input_dim)
    
    x = initial_conv(ip)
    nb_conv = 4

    x = expand_conv(x, 16, k)
    nb_conv += 2

    for i in range(N - 1):
        x = conv1_block(x, k, dropout)
        nb_conv += 2

    x = BatchNormalization(axis=channel_axis, momentum=mom,
                           epsilon=eps, gamma_initializer=gamma)(x)
    x = LeakyReLU(leakiness)(x)

    x = expand_conv(x, 32, k, strides=(1, 1))
    x = MaxPooling2D((2, 2), padding='same')(x)
    nb_conv += 2

    for i in range(N - 1):
        x = conv2_block(x, k, dropout)
        nb_conv += 2

    x = BatchNormalization(axis=channel_axis, momentum=mom,
                           epsilon=eps, gamma_initializer=gamma)(x)
    x = LeakyReLU(leakiness)(x)

    x = expand_conv(x, 64, k, strides=(1, 1))
    x = MaxPooling2D((2, 2), padding='same')(x)
    nb_conv += 2

    for i in range(N - 1):
        x = conv3_block(x, k, dropout)
        nb_conv += 2

    x = BatchNormalization(axis=channel_axis, momentum=mom,
                           epsilon=eps, gamma_initializer=gamma)(x)
    x = LeakyReLU(leakiness)(x)
    
    # Return output dimensions 8 x 8 x 128
    model = Model(ip, x, name='wrn_trunk')
    print("Wide Residual Network-%d-%d created." % (nb_conv, k))
    return model

