from keras.optimizers import Adam
import glob

data_dir = "data/*.*"
test_dir = "test/*.*"
test_results_dir = "test_results/"

# Make a list of all images inside the data directory
ALL_IMAGES = glob.glob(data_dir)
TEST_IMAGES = glob.glob(test_dir)
TEST_IMAGES_RESULT = test_results_dir

# Shape of low-resolution and higher-resolution images
low_resolution_shape = (64, 64, 3)
higher_resolution_shape = (256, 256, 3)

# Adam optimizer used for all networks
common_optimizer = Adam(0.0002, 0.5)
