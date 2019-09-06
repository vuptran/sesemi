"""Max-Pooling Network-in-Network with BatchNormalization."""

from keras import initializers
from keras.models import Model
from keras.regularizers import l2
from keras.layers import BatchNormalization
from keras.layers import ZeroPadding2D, MaxPooling2D
from keras.layers import Input, Conv2D, Dropout, LeakyReLU

leakiness = 0.0
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
        padding='valid',
        kernel_initializer=initer,
        kernel_regularizer=l2(weight_decay),
    )


def create_network(input_shape, dropout=0.0):
    data = Input(shape=input_shape) 
    
    x = ZeroPadding2D(padding=(2, 2))(data)
    x = Conv2D(192, (5, 5), **conv_params)(x)
    x = BatchNormalization(**bn_params)(x)
    x = LeakyReLU(leakiness)(x)

    x = Conv2D(160, (1, 1), **conv_params)(x)
    x = BatchNormalization(**bn_params)(x)
    x = LeakyReLU(leakiness)(x)

    x = Conv2D(96, (1, 1), **conv_params)(x)
    x = BatchNormalization(**bn_params)(x)
    x = LeakyReLU(leakiness)(x)

    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    if dropout > 0.0: x = Dropout(dropout)(x)

    x = ZeroPadding2D(padding=(2, 2))(x)
    x = Conv2D(192, (5, 5), **conv_params)(x)
    x = BatchNormalization(**bn_params)(x)
    x = LeakyReLU(leakiness)(x)

    x = Conv2D(192, (1, 1), **conv_params)(x)
    x = BatchNormalization(**bn_params)(x)
    x = LeakyReLU(leakiness)(x)

    x = Conv2D(192, (1, 1), **conv_params)(x)
    x = BatchNormalization(**bn_params)(x)
    x = LeakyReLU(leakiness)(x)

    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    if dropout > 0.0: x = Dropout(dropout)(x)

    x = ZeroPadding2D(padding=(1, 1))(x)
    x = Conv2D(192, (3, 3), **conv_params)(x)
    x = BatchNormalization(**bn_params)(x)
    x = LeakyReLU(leakiness)(x)

    x = Conv2D(192, (1, 1), **conv_params)(x)
    x = BatchNormalization(**bn_params)(x)
    x = LeakyReLU(leakiness)(x)

    x = Conv2D(192, (1, 1), **conv_params)(x)
    x = BatchNormalization(**bn_params)(x)
    x = LeakyReLU(leakiness)(x)

    # Return output dimensions 8 x 8 x 192
    net = Model(data, x, name='nin_trunk')
    return net

