from scipy.misc import imread, imresize
import numpy as np
import matplotlib.pyplot as plt

from config import TEST_IMAGES, TEST_IMAGES_RESULT
from generator import build_generator


def main():
    # Build the generator network
    generator = build_generator()

    # Load model weights
    generator.load_weights("generator.h5")

    low_resolution_images = list()
    filename = "failed_image.jpg"

    for img in TEST_IMAGES:
        filename = img
        img1 = imread(img, mode="RGB")
        img1 = img1.astype(np.float32)

        # Resize the image
        img1_low_resolution = imresize(img1, (64, 64, 3))

        low_resolution_images.append(img1_low_resolution)

    low_resolution_images = np.array(low_resolution_images) / 127.5 - 1.0

    # Generate high-resolution images from low-resolution images
    generated_images = generator.predict_on_batch(low_resolution_images)

    # Save image
    for index, img in enumerate(generated_images):
        fig = plt.figure(figsize=(16, 9))
        ax = fig.add_subplot(1, 2, 1)
        ax.imshow(low_resolution_images[0], interpolation="nearest")
        ax.axis("off")
        ax.set_title("Low-resolution")

        ax = fig.add_subplot(1, 2, 2)
        ax.imshow(img, interpolation="nearest")
        ax.axis("off")
        ax.set_title("Generated")

        plt.savefig("{0}{1}".format(TEST_IMAGES_RESULT, filename.split("/")[-1]))


if __name__ == "__main__":
    main()




"""
if mode == "predict":
    # Build and compile the discriminator network
    discriminator = build_discriminator()

    # Build the generator network
    generator = build_generator()

    # Load models
    generator.load_weights("generator.h5")
    discriminator.load_weights("discriminator.h5")

    # Get 10 random images
    high_resolution_images, low_resolution_images = sample_images(
        data_dir=data_dir,
        batch_size=1,
        low_resolution_shape=low_resolution_shape,
        high_resolution_shape=high_resolution_shape,
    )
    # Normalize images
    high_resolution_images = high_resolution_images / 127.5 - 1.0
    low_resolution_images = low_resolution_images / 127.5 - 1.0

    # Generate high-resolution images from low-resolution images
    generated_images = generator.predict_on_batch(low_resolution_images)

    # Save images
    for index, img in enumerate(generated_images):
        save_images(
            low_resolution_images[index],
            high_resolution_images[index],
            img,
            path="predict_results/gen_{}".format(index),
        )

"""