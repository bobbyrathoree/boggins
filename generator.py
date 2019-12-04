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
    Create a generator network using the hyper-parameter values defined below
    :return:
    """
    residual_blocks = 16
    momentum = 0.8
    input_shape = (64, 64, 3)

    # Input Layer of the generator network
    input_layer = Input(shape=input_shape)

    # The pre-residual block
    generator_one = Conv2D(
        filters=64, kernel_size=9, strides=1, padding="same", activation="relu"
    )(input_layer)

    # Add 16 skip-connection blocks
    residual = residual_block(generator_one)
    for _ in range(residual_blocks - 1):
        residual = residual_block(residual)

    # The post-residual block
    generator_two = Conv2D(filters=64, kernel_size=3, strides=1, padding="same")(
        residual
    )
    generator_two = BatchNormalization(momentum=momentum)(generator_two)

    # Take the sum of the output from the pre-residual block(gen1) and the post-residual block(gen2)
    generator_three = Add()([generator_two, generator_one])

    # Add an upsampling block
    generator_four = UpSampling2D(size=2)(generator_three)
    generator_four = Conv2D(filters=256, kernel_size=3, strides=1, padding="same")(
        generator_four
    )
    generator_four = Activation("relu")(generator_four)

    # Add another upsampling block
    generator_five = UpSampling2D(size=2)(generator_four)
    generator_five = Conv2D(filters=256, kernel_size=3, strides=1, padding="same")(
        generator_five
    )
    generator_five = Activation("relu")(generator_five)

    # Output convolution layer
    generator_six = Conv2D(filters=3, kernel_size=9, strides=1, padding="same")(
        generator_five
    )
    output = Activation("tanh")(generator_six)

    # Final generator model
    model = Model(inputs=[input_layer], outputs=[output], name="generator")
    return model
