from keras.optimizers import Adam
import glob

data_dir = "data/img_align_celeba/*.*"
epochs = 50000
batch_size = 5
mode = "train"

# Make a list of all images inside the data directory
ALL_IMAGES = glob.glob(data_dir)

# Shape of low-resolution and high-resolution images
low_resolution_shape = (64, 64, 3)
high_resolution_shape = (256, 256, 3)

# Common optimizer for all networks
common_optimizer = Adam(0.0002, 0.5)
