from keras.optimizers import Adam
import glob

data_dir = "data/img_align_celeba/*.*"
test_dir = "test/*.*"
test_results_dir = "test_results/"
epochs = 50000
batch_size = 5
mode = "predict"

# Make a list of all images inside the data directory
ALL_IMAGES = glob.glob(data_dir)
TEST_IMAGES = glob.glob(test_dir)
TEST_IMAGES_RESULT = test_results_dir

# Shape of low-resolution and high-resolution images
low_resolution_shape = (64, 64, 3)
high_resolution_shape = (256, 256, 3)

# Common optimizer for all networks
common_optimizer = Adam(0.0002, 0.5)
