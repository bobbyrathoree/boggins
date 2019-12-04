import time
from argparse import ArgumentParser

import numpy as np
from keras import Input
from keras.callbacks import TensorBoard
from keras.models import Model
from discriminator import build_discriminator
from generator import build_generator
from utils import (
    sample_images,
    write_log,
    save_images,
    common_optimizer,
    higher_resolution_shape,
    low_resolution_shape,
)
from vgg import build_vgg


def train(epochs: int, batch_size: int):
    """
    Main method to train boggins
    :param epochs: number of epochs to train for
    :param batch_size: batch size
    """
    # Build VGG network
    vgg = build_vgg()

    # We don't need to train it, make it non-trainable
    vgg.trainable = False

    # Compile VGG19 network to extract features maps
    vgg.compile(loss="mse", optimizer=common_optimizer, metrics=["accuracy"])

    # Build and compile the discriminator network
    discriminator = build_discriminator()
    discriminator.compile(loss="mse", optimizer=common_optimizer, metrics=["accuracy"])

    # Build the generator network
    generator = build_generator()

    # Input layers for better-resolution and low-resolution images
    input_better_resolution = Input(shape=higher_resolution_shape)
    input_low_resolution = Input(shape=low_resolution_shape)

    # Generate better-resolution images from low-resolution images
    generated_better_resolution_images = generator(input_low_resolution)

    # Extract feature maps of the generated images
    features = vgg(generated_better_resolution_images)

    # Set the discriminator network as non-trainable
    discriminator.trainable = False

    # Get the probability of generated better-resolution images
    probabilities = discriminator(generated_better_resolution_images)

    # Create and compile an adversarial model
    adversarial_model = Model(
        [input_low_resolution, input_better_resolution], [probabilities, features]
    )
    adversarial_model.compile(
        loss=["binary_crossentropy", "mse"],
        loss_weights=[1e-3, 1],
        optimizer=common_optimizer,
    )

    # Add TensorBoard callback
    tb_callback_obj = TensorBoard(log_dir="logs/".format(time.time()))
    tb_callback_obj.set_model(generator)
    tb_callback_obj.set_model(discriminator)

    for epoch in range(epochs):
        print("Epoch: {0}".format(epoch))

        # Sample a batch of images for training the discriminator network
        better_resolution_images, low_resolution_images = sample_images(
            batch_size=batch_size,
            low_resolution_shape=low_resolution_shape,
            higher_resolution_shape=higher_resolution_shape,
        )

        # Normalize images to train upon
        better_resolution_images = better_resolution_images / 127.5 - 1.0
        low_resolution_images = low_resolution_images / 127.5 - 1.0

        # Generate better-resolution images from low-resolution images
        generated_better_resolution_images = generator.predict(low_resolution_images)

        # Generate batch of real and fake labels
        real_tokens = np.ones((batch_size, 16, 16, 1))
        fake_tokens = np.zeros((batch_size, 16, 16, 1))

        # Train the discriminator network on real and fake images
        discriminator_loss_real = discriminator.train_on_batch(
            better_resolution_images, real_tokens
        )
        discriminator_loss_fake = discriminator.train_on_batch(
            generated_better_resolution_images, fake_tokens
        )

        # Calculate total discriminator loss
        discriminator_loss = 0.5 * np.add(
            discriminator_loss_real, discriminator_loss_fake
        )
        print("discriminator_loss: ", discriminator_loss)

        # Sample a batch of images for training the generator network
        better_resolution_images, low_resolution_images = sample_images(
            batch_size=batch_size,
            low_resolution_shape=low_resolution_shape,
            higher_resolution_shape=higher_resolution_shape,
        )

        # Normalize images
        better_resolution_images = better_resolution_images / 127.5 - 1.0
        low_resolution_images = low_resolution_images / 127.5 - 1.0

        # Extract feature maps for real better-resolution images
        feature_maps_of_image = vgg.predict(better_resolution_images)

        # Train the generator network
        generator_loss = adversarial_model.train_on_batch(
            [low_resolution_images, better_resolution_images],
            [real_tokens, feature_maps_of_image],
        )

        print("generator_loss: ", generator_loss)

        # Write the losses to Tensorboard
        write_log(tb_callback_obj, "g_loss", generator_loss[0], epoch)
        write_log(tb_callback_obj, "d_loss", discriminator_loss[0], epoch)

        # Sample and save images after every 100 epochs
        if epoch % 100 == 0:
            better_resolution_images, low_resolution_images = sample_images(
                batch_size=batch_size,
                low_resolution_shape=low_resolution_shape,
                higher_resolution_shape=higher_resolution_shape,
            )

            # Normalize images
            better_resolution_images = better_resolution_images / 127.5 - 1.0
            low_resolution_images = low_resolution_images / 127.5 - 1.0

            generated_images = generator.predict_on_batch(low_resolution_images)

            for index, img in enumerate(generated_images):
                save_images(
                    low_resolution_images[index],
                    better_resolution_images[index],
                    img,
                    path="training_results/img_{}_{}".format(epoch, index),
                )

    # When training complete, save the models in the local directory
    generator.save_weights("models/generator.h5")
    discriminator.save_weights("models/discriminator.h5")


if __name__ == "__main__":

    # Add option to provide number of epochs and batch size
    parser = ArgumentParser()
    parser.add_argument("--epochs", "-e", type=int, default=50000, required=False)
    parser.add_argument("--batch", "-b", type=int, default=32, required=False)
    args = parser.parse_args()

    train(epochs=args.epochs, batch_size=args.batch)
