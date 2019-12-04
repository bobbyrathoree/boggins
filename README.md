# Boggins AI: Vision Enhancer
SRGAN to generate a better resolution version of a low-res image.

Datasets used to train: [CelebA](https://bit.ly/380hp3K) and [imdb-wiki](https://bit.ly/2LaAMxp)

Clone our project using: `git clone https://github.com/bobbyrathoree/boggins.git`

Create an environment using conda or venv, activate it and run: `pip install -r requirements.txt`

### Train
1. Create data directory: `mkdir data`
2. Unzip the dataset files (all images) into data directory.
3. Run: `python train.py --epochs <desired_epochs> --batch <desired_batch_size>`

**Note:** If epochs and batch are not set, the model trains for 50000 epochs using a batch size of 32.

### Test
1. Run: `python test.py --input <path-to-your-image-file>`

**Note:** The file is saved in test_results directory.

For training and testing results, visit the [website](https://boggins.firebaseapp.com/).
