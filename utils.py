import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import imread, imresize
import tensorflow as tf


def sample_images(data_dir, batch_size, high_resolution_shape, low_resolution_shape):
    # Make a list of all images inside the data directory
    # all_images = glob.glob(data_dir)

    # Choose a random batch of images
    images_batch = np.random.choice(ALL_IMAGES, size=batch_size)

    low_resolution_images = []
    high_resolution_images = []
    try:
        for img in images_batch:
            # Get an ndarray of the current image
            img1 = imread(img, mode="RGB")
            img1 = img1.astype(np.float32)

            # Resize the image
            img1_high_resolution = imresize(img1, high_resolution_shape)
            img1_low_resolution = imresize(img1, low_resolution_shape)

            # Do a random horizontal flip
            if np.random.random() < 0.5:
                img1_high_resolution = np.fliplr(img1_high_resolution)
                img1_low_resolution = np.fliplr(img1_low_resolution)

            high_resolution_images.append(img1_high_resolution)
            low_resolution_images.append(img1_low_resolution)
    except TypeError as e:
        return sample_images(
            data_dir, batch_size, high_resolution_shape, low_resolution_shape
        )

    # Convert the lists to Numpy NDArrays
    return np.array(high_resolution_images), np.array(low_resolution_images)


def save_images(low_resolution_image, original_image, generated_image, path):
    """
    Save low-resolution, high-resolution(original) and
    generated high-resolution images in a single image
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
    Write scalars to Tensorboard
    """
    summary = tf.Summary()
    summary_value = summary.value.add()
    summary_value.simple_value = value
    summary_value.tag = name
    callback.writer.add_summary(summary, batch_no)
    callback.writer.flush()
