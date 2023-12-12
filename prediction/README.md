# Prediction Folder

`predict.py` is used for predicting 3D semantic segmentation using a pre-built 3D U-Net model.

## Usage - `predict.py`

To use the `predict.py` script, follow these steps:

1. Open a terminal or command prompt.

2. Run the `predict.py` script with the following command-line arguments:

   - `modelfile`: Model filename, specifying the trained 3D U-Net model you want to use for prediction.
   - `inputfile`: Input filename, specifying the input data or image you want to perform semantic segmentation on.
   - `outputfile`: Output filename, specifying where you want to save the segmentation results.

   For example:

   ```bash
   python predict.py random_block_sampling_v4_augmentations3000.h5 Cell1_2_Crop1-1.tif Cell1_2_Crop1-1
