"""13-Layer Max-Pooling ConvNet with BatchNormalization."""

from keras import initializers
from keras.models import Model
from keras.regularizers import l2
from keras.layers import MaxPooling2D, BatchNormalization
from keras.layers import Input, Conv2D, Dropout, LeakyReLU

leakiness = 0.1
weight_decay = 0.0005
initer = initializers.he_normal()

bn_params = dict(
        axis=-1,
        momentum=0.99,
        epsilon=0.001,
        gamma_initializer='ones',
    )

conv_params = dict(
        use_bias=True,
        kernel_initializer=initer,
        kernel_regularizer=l2(weight_decay),
    )


def create_network(input_shape, dropout=0.0):
    data = Input(shape=input_shape) 

    x = Conv2D(128, (3, 3), padding='same', **conv_params)(data)
    x = BatchNormalization(**bn_params)(x)
    x = LeakyReLU(leakiness)(x)

    x = Conv2D(128, (3, 3), padding='same', **conv_params)(x)
    x = BatchNormalization(**bn_params)(x)
    x = LeakyReLU(leakiness)(x)

    x = Conv2D(128, (3, 3), padding='same', **conv_params)(x)
    x = BatchNormalization(**bn_params)(x)
    x = LeakyReLU(leakiness)(x)

    x = MaxPooling2D((2, 2), strides=(2, 2), padding='valid')(x)
    if dropout > 0.0: x = Dropout(dropout)(x)

    x = Conv2D(256, (3, 3), padding='same', **conv_params)(x)
    x = BatchNormalization(**bn_params)(x)
    x = LeakyReLU(leakiness)(x)

    x = Conv2D(256, (3, 3), padding='same', **conv_params)(x)
    x = BatchNormalization(**bn_params)(x)
    x = LeakyReLU(leakiness)(x)

    x = Conv2D(256, (3, 3), padding='same', **conv_params)(x)
    x = BatchNormalization(**bn_params)(x)
    x = LeakyReLU(leakiness)(x)

    x = MaxPooling2D((2, 2), strides=(2, 2), padding='valid')(x)
    if dropout > 0.0: x = Dropout(dropout)(x)

    x = Conv2D(512, (3, 3), padding='valid', **conv_params)(x)
    x = BatchNormalization(**bn_params)(x)
    x = LeakyReLU(leakiness)(x)

    x = Conv2D(256, (1, 1), padding='valid', **conv_params)(x)
    x = BatchNormalization(**bn_params)(x)
    x = LeakyReLU(leakiness)(x)

    x = Conv2D(128, (1, 1), padding='valid', **conv_params)(x)
    x = BatchNormalization(**bn_params)(x)
    x = LeakyReLU(leakiness)(x)

    # Return output dimensions 6 x 6 x 128
    net = Model(data, x, name='convnet_trunk')
    return net

