from keras import Input
from keras.layers import BatchNormalization, LeakyReLU, Dense
from keras.layers.convolutional import Conv2D
from keras.models import Model


def build_discriminator():
    """
    Create a discriminator network using the hyper-parameter values defined below
    :return:
    """
    leakyrelu_alpha = 0.2
    momentum = 0.8
    input_shape = (256, 256, 3)

    input_layer = Input(shape=input_shape)

    # Add the first convolution block
    discriminator_one = Conv2D(filters=64, kernel_size=3, strides=1, padding="same")(
        input_layer
    )
    discriminator_one = LeakyReLU(alpha=leakyrelu_alpha)(discriminator_one)

    # Add the 2nd convolution block
    discriminator_two = Conv2D(filters=64, kernel_size=3, strides=2, padding="same")(
        discriminator_one
    )
    discriminator_two = LeakyReLU(alpha=leakyrelu_alpha)(discriminator_two)
    discriminator_two = BatchNormalization(momentum=momentum)(discriminator_two)

    # Add the third convolution block
    discriminator_three = Conv2D(filters=128, kernel_size=3, strides=1, padding="same")(
        discriminator_two
    )
    discriminator_three = LeakyReLU(alpha=leakyrelu_alpha)(discriminator_three)
    discriminator_three = BatchNormalization(momentum=momentum)(discriminator_three)

    # Add the fourth convolution block
    discriminator_four = Conv2D(filters=128, kernel_size=3, strides=2, padding="same")(
        discriminator_three
    )
    discriminator_four = LeakyReLU(alpha=leakyrelu_alpha)(discriminator_four)
    discriminator_four = BatchNormalization(momentum=0.8)(discriminator_four)

    # Add the fifth convolution block
    discriminator_five = Conv2D(256, kernel_size=3, strides=1, padding="same")(
        discriminator_four
    )
    discriminator_five = LeakyReLU(alpha=leakyrelu_alpha)(discriminator_five)
    discriminator_five = BatchNormalization(momentum=momentum)(discriminator_five)

    # Add the sixth convolution block
    discriminator_six = Conv2D(filters=256, kernel_size=3, strides=2, padding="same")(
        discriminator_five
    )
    discriminator_six = LeakyReLU(alpha=leakyrelu_alpha)(discriminator_six)
    discriminator_six = BatchNormalization(momentum=momentum)(discriminator_six)

    # Add the seventh convolution block
    discriminator_seven = Conv2D(filters=512, kernel_size=3, strides=1, padding="same")(
        discriminator_six
    )
    discriminator_seven = LeakyReLU(alpha=leakyrelu_alpha)(discriminator_seven)
    discriminator_seven = BatchNormalization(momentum=momentum)(discriminator_seven)

    # Add the eight convolution block
    discriminator_eight = Conv2D(filters=512, kernel_size=3, strides=2, padding="same")(
        discriminator_seven
    )
    discriminator_eight = LeakyReLU(alpha=leakyrelu_alpha)(discriminator_eight)
    discriminator_eight = BatchNormalization(momentum=momentum)(discriminator_eight)

    # Add a dense layer
    discriminator_nine_dense = Dense(units=1024)(discriminator_eight)
    discriminator_nine_dense = LeakyReLU(alpha=0.2)(discriminator_nine_dense)

    # Last dense layer - for classifying how close generated image is to the original
    output = Dense(units=1, activation="sigmoid")(discriminator_nine_dense)

    model = Model(inputs=[input_layer], outputs=[output], name="discriminator")
    return model
