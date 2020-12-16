import glob
import sys
import os
from PIL import Image


def get_all_image_files(directory):
    image_files = list()
    for files in ("*.png", "*.jpg"):
        image_files.extend(glob.glob("{0}/{1}".format(directory, files)))
    return image_files


if __name__ == "__main__":
    os.makedirs("{0}/resize_results".format(sys.argv[1]))
    for file in get_all_image_files(sys.argv[1]):
        image_pil = Image.open(file)
        resized_image_pil = image_pil.resize(
            (int(image_pil.size[0] / 6), int(image_pil.size[1] / 6)), Image.ANTIALIAS
        )
        print(
            "Original image size: {0}\tNew size: {1}".format(
                image_pil.size, resized_image_pil.size
            )
        )
        resized_image_pil.save("{0}/resize_results/{1}".format(sys.argv[1], file.split("/")[-1]))
