import time

import numpy as np
from keras import Input
from keras.callbacks import TensorBoard
from keras.models import Model
from config import *
from discriminator import build_discriminator
from generator import build_generator
from utils import sample_images, write_log, save_images
from vgg import build_vgg


def train():
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

    # Input layers for high-resolution and low-resolution images
    input_high_resolution = Input(shape=high_resolution_shape)
    input_low_resolution = Input(shape=low_resolution_shape)

    # Generate high-resolution images from low-resolution images
    generated_high_resolution_images = generator(input_low_resolution)

    # Extract feature maps of the generated images
    features = vgg(generated_high_resolution_images)

    # Make the discriminator network as non-trainable
    discriminator.trainable = False

    # Get the probability of generated high-resolution images
    probabilities = discriminator(generated_high_resolution_images)

    # Create and compile an adversarial model
    adversarial_model = Model(
        [input_low_resolution, input_high_resolution], [probabilities, features]
    )
    adversarial_model.compile(
        loss=["binary_crossentropy", "mse"],
        loss_weights=[1e-3, 1],
        optimizer=common_optimizer,
    )

    # Add TensorBoard callback
    tensorboard = TensorBoard(log_dir="logs/".format(time.time()))
    tensorboard.set_model(generator)
    tensorboard.set_model(discriminator)

    for epoch in range(epochs):
        print("Epoch: {0}".format(epoch))

        # Sample a batch of images for training the discriminator network
        high_resolution_images, low_resolution_images = sample_images(
            batch_size=batch_size,
            low_resolution_shape=low_resolution_shape,
            high_resolution_shape=high_resolution_shape,
        )

        # Normalize images
        high_resolution_images = high_resolution_images / 127.5 - 1.0
        low_resolution_images = low_resolution_images / 127.5 - 1.0

        # Generate high-resolution images from low-resolution images
        generated_high_resolution_images = generator.predict(low_resolution_images)

        # Generate batch of real and fake labels
        real_labels = np.ones((batch_size, 16, 16, 1))
        fake_labels = np.zeros((batch_size, 16, 16, 1))

        # Train the discriminator network on real and fake images
        d_loss_real = discriminator.train_on_batch(high_resolution_images, real_labels)
        d_loss_fake = discriminator.train_on_batch(
            generated_high_resolution_images, fake_labels
        )

        # Calculate total discriminator loss
        discriminator_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        print("discriminator_loss: ", discriminator_loss)

        # Sample a batch of images for training the generator network
        high_resolution_images, low_resolution_images = sample_images(
            batch_size=batch_size,
            low_resolution_shape=low_resolution_shape,
            high_resolution_shape=high_resolution_shape,
        )

        # Normalize images
        high_resolution_images = high_resolution_images / 127.5 - 1.0
        low_resolution_images = low_resolution_images / 127.5 - 1.0

        # Extract feature maps for real high-resolution images
        image_features = vgg.predict(high_resolution_images)

        # Train the generator network
        generator_loss = adversarial_model.train_on_batch(
            [low_resolution_images, high_resolution_images],
            [real_labels, image_features],
        )

        print("generator_loss: ", generator_loss)

        # Write the losses to Tensorboard
        write_log(tensorboard, "g_loss", generator_loss[0], epoch)
        write_log(tensorboard, "d_loss", discriminator_loss[0], epoch)

        # Sample and save images after every 100 epochs
        if epoch % 100 == 0:
            high_resolution_images, low_resolution_images = sample_images(
                batch_size=batch_size,
                low_resolution_shape=low_resolution_shape,
                high_resolution_shape=high_resolution_shape,
            )

            # Normalize images
            high_resolution_images = high_resolution_images / 127.5 - 1.0
            low_resolution_images = low_resolution_images / 127.5 - 1.0

            generated_images = generator.predict_on_batch(low_resolution_images)

            for index, img in enumerate(generated_images):
                save_images(
                    low_resolution_images[index],
                    high_resolution_images[index],
                    img,
                    path="results/img_{}_{}".format(epoch, index),
                )

    # Save models
    generator.save_weights("generator.h5")
    discriminator.save_weights("discriminator.h5")


if __name__ == "__main__":
    train()
