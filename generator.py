from keras import Input
from keras.layers import BatchNormalization, Activation, Add
from keras.layers.convolutional import Conv2D, UpSampling2D
from keras.models import Model


def residual_block(layer):
    """
    A skip-connection residual block.
    :param layer: input layer
    :return:
    """
    filters = [64, 64]
    kernel_size = 3
    strides = 1
    padding = "same"
    momentum = 0.8
    activation = "relu"

    residual = Conv2D(
        filters=filters[0], kernel_size=kernel_size, strides=strides, padding=padding
    )(layer)
    residual = Activation(activation=activation)(residual)
    residual = BatchNormalization(momentum=momentum)(residual)

    residual = Conv2D(
        filters=filters[1], kernel_size=kernel_size, strides=strides, padding=padding
    )(residual)
    residual = BatchNormalization(momentum=momentum)(residual)

    # Add residual and layer
    residual = Add()([residual, layer])
    return residual


def build_generator():
    """
    Create a generator network using the hyperparameter values defined below
    :return:
    """
    residual_blocks = 16
    momentum = 0.8
    input_shape = (64, 64, 3)

    # Input Layer of the generator network
    input_layer = Input(shape=input_shape)

    # The pre-residual block
    gen1 = Conv2D(
        filters=64, kernel_size=9, strides=1, padding="same", activation="relu"
    )(input_layer)

    # Add 16 skip-connection blocks
    res = residual_block(gen1)
    for i in range(residual_blocks - 1):
        res = residual_block(res)

    # The post-residual block
    gen2 = Conv2D(filters=64, kernel_size=3, strides=1, padding="same")(res)
    gen2 = BatchNormalization(momentum=momentum)(gen2)

    # Take the sum of the output from the pre-residual block(gen1) and the post-residual block(gen2)
    gen3 = Add()([gen2, gen1])

    # Add an upsampling block
    gen4 = UpSampling2D(size=2)(gen3)
    gen4 = Conv2D(filters=256, kernel_size=3, strides=1, padding="same")(gen4)
    gen4 = Activation("relu")(gen4)

    # Add another upsampling block
    gen5 = UpSampling2D(size=2)(gen4)
    gen5 = Conv2D(filters=256, kernel_size=3, strides=1, padding="same")(gen5)
    gen5 = Activation("relu")(gen5)

    # Output convolution layer
    gen6 = Conv2D(filters=3, kernel_size=9, strides=1, padding="same")(gen5)
    output = Activation("tanh")(gen6)

    # Final generator model
    model = Model(inputs=[input_layer], outputs=[output], name="generator")
    return model
