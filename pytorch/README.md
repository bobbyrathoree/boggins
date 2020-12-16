# Boggins+

### Testing
To test the model, you'll need to download my pretrained model from here. Copy the downloaded .pth file's location and run this command:

`python test.py <path-to-pth-file>`

This command runs the model on sample low-resolution images present in the data/test/LR directory. The results will be saved in data/test/results directory.

I encourage you to add your own images to the source directory (data/test/LR) and view their results. If you dont have low-res images to test on, use the helper script _make_low_res.py_ to create them from  your high-res images.

`python make_low_res.py <path-to-high-res-images-directory>`

The script will save the new low-res images in the same directory in a _resize_results_ directory. Copy these images to data/test/LR and run the test script.


### Training
To train your own model, you'll need to download the datasets using the fetch_dataset.sh script. After download is finished, you'll need to change the dataroot parameters in training configuration file _train.json_ according to where the datasets were now downloaded. Once you have training options file configured, run:

`python train.py`