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


# Creating an AVI File from Output TIF

After using `predict.py` for 3D semantic segmentation with a pre-built 3D U-Net model, follow these steps to convert the output TIF file into an AVI file. This process is designed for users who have completed semantic segmentation and wish to create a video representation of their results.

## Prerequisites

Ensure you have a Conda environment set up with all necessary dependencies installed as specified in `requirements.txt`. If you haven't done this yet, please refer to the section on setting up your environment before proceeding.

## Steps

1. **Activate Conda Environment**: Open a terminal or command prompt and activate the Conda environment where you have installed all the required packages. Use the command:
   
   ```bash
   conda activate your_environment_name

2. **Navigate to Prediction Directory**: Use the `cd` command to navigate to the directory containing the `predict.py` script. For example:
   ```bash
   cd path/to/prediction_directory

3. **Run Prediction Command**: Execute the prediction script with the necessary command-line arguments. For instance:
   ```bash
   python predict.py modelfile.h5 inputfile.tif outputfile\

4. **Save Largest TIF File**: After the prediction process, locate the largest TIF file generated (usually the one containing the complete segmented output) and move it to an easily accessible directory.

5. **Open Fiji**: Launch the Fiji application. Fiji is an image processing package—a "batteries-included" distribution of ImageJ, including many plugins which facilitate scientific image analysis.

6. **Open TIF File in Fiji**: In Fiji, navigate to `File` -> `Open`, and select the TIF file you moved in the previous step.

7. **Convert to 3D Projection**:
- Within Fiji, go to `Image` -> `Stacks` -> `3D Project`.
- Adjust the parameters as necessary for your specific visualization needs and click `OK` to generate the 3D projection.

8. **Save as AVI**:
- Navigate to `File` -> `Save As` -> `AVI...`.
- Choose the desired location, name your file, and select `Save`.

9. **Optional - No Compression for Best Quality**:
- When saving as AVI, you may have the option to save with no compression. This is recommended for maintaining the best quality, although it will result in a larger file size.

By following these steps, you can successfully create an AVI file from the output TIF file, providing a dynamic visualization of your 3D semantic segmentation results.

