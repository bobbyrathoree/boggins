from argparse import ArgumentParser

from scipy.misc import imread, imresize
import numpy as np
import matplotlib.pyplot as plt

from config import TEST_IMAGES_RESULT, low_resolution_shape
from generator import build_generator


def test(file_path: str):
    """
    Function to enhance a low resolution image and save it alongside its enhanced version
    :param file_path: path of input file
    """
    # Build the generator network
    generator = build_generator()

    # Load model weights
    generator.load_weights("models/generator.h5")

    # Make a low-res counterpart to predict upon
    image = imread(file_path, mode="RGB")
    image = image.astype(np.float32)
    low_resolution_image = [imresize(image, low_resolution_shape)]

    low_resolution_image = np.array(low_resolution_image) / 127.5 - 1.0

    # Generate high-resolution images from low-resolution images
    generated_image = generator.predict_on_batch(low_resolution_image)

    # Make a common figure
    fig = plt.figure(figsize=(16, 9))

    # Add the low resolution image on the left side
    ax = fig.add_subplot(1, 2, 1)
    ax.imshow(low_resolution_image[0], interpolation="nearest")
    ax.axis("off")
    ax.set_title("Low-resolution")

    # Add generated image to figure on the right side
    ax = fig.add_subplot(1, 2, 2)
    ax.imshow(generated_image[0], interpolation="nearest")
    ax.axis("off")
    ax.set_title("Generated")

    # Save the figure
    plt.savefig("{0}{1}".format(TEST_IMAGES_RESULT, file_path.split("/")[-1]))


if __name__ == "__main__":

    # Add option to provide input file path
    parser = ArgumentParser()
    parser.add_argument("--input", "-i", type=str, required=True)
    args = parser.parse_args()

    test(file_path=args.input)
