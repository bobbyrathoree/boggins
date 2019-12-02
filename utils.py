import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import imread, imresize
import tensorflow as tf
from config import *


def sample_images(batch_size, higher_resolution_shape, low_resolution_shape):
    """
    :param data_dir:
    :param batch_size:
    :param higher_resolution_shape:
    :param low_resolution_shape:
    :return:
    """
    # Choose a random batch of images
    images_batch = np.random.choice(ALL_IMAGES, size=batch_size)

    low_resolution_images = []
    high_resolution_images = []

    try:
        for img in images_batch:

            # Read the image in RGB mode
            sample_image = imread(img, mode="RGB")
            sample_image = sample_image.astype(np.float32)

            # Resize the image
            img1_high_resolution = imresize(sample_image, higher_resolution_shape)
            img1_low_resolution = imresize(sample_image, low_resolution_shape)

            # Do a flip sometimes
            if np.random.random() < 0.5:
                img1_high_resolution = np.fliplr(img1_high_resolution)
                img1_low_resolution = np.fliplr(img1_low_resolution)

            high_resolution_images.append(img1_high_resolution)
            low_resolution_images.append(img1_low_resolution)

    except TypeError as e:
        return sample_images(batch_size, higher_resolution_shape, low_resolution_shape)

    return np.array(high_resolution_images), np.array(low_resolution_images)


def save_images(low_resolution_image, original_image, generated_image, path):
    """
    Save low-resolution, original and generated super-resolution images in a single image
    :param low_resolution_image:
    :param original_image:
    :param generated_image:
    :param path:
    """
    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot(1, 3, 1)
    ax.imshow(low_resolution_image, interpolation="nearest")
    ax.axis("off")
    ax.set_title("Low-resolution")

    ax = fig.add_subplot(1, 3, 2)
    ax.imshow(original_image, interpolation="nearest")
    ax.axis("off")
    ax.set_title("Original")

    ax = fig.add_subplot(1, 3, 3)
    ax.imshow(generated_image, interpolation="nearest")
    ax.axis("off")
    ax.set_title("Generated")

    plt.savefig(path)


def write_log(callback, name, value, batch_no):
    """
    Write losses to Tensorboard
    :param callback:
    :param name:
    :param value:
    :param batch_no:
    """
    summary = tf.Summary()
    summary_value = summary.value.add()
    summary_value.simple_value = value
    summary_value.tag = name
    callback.writer.add_summary(summary, batch_no)
    callback.writer.flush()
