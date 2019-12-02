# Boggins AI: Vision Enhancer
SRGAN to generate a better resolution version of a low-res image.

Datasets used to train: [CelebA](https://bit.ly/380hp3K) and [imdb-wiki](https://bit.ly/2LaAMxp)

### Train
1. Unzip the dataset files (all images) into data directory.
2. Run: `python train.py --epochs <desired_epochs> --batch <desired_batch_size>`

**Note:** If epochs and batch are not set, the model trains for 50000 epochs using a batch size of 32.

### Test
1. Run: `python test.py --input <path-to-your-image-file>`

**Note:** The file is saved in test_results directory.