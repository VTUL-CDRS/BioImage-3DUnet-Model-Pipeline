# BioImage 3DUnet Model Pipeline

BioImage 3DUnet Model Pipeline is a data curation pipeline designed for processing very large biomedical images. It includes AI-assisted human-in-the-loop segmentation of nanoscale images using 3DUnet deep learning.

# Contents

/postprocessing: Contains tools for preprocessing raw .tif files into the desired shape for training and prediction.

/training: Includes the main training programs. The files here are used to train models using specified raw files and their labels.

/prediction: Contains programs for making predictions using specified models on raw files. Predictions are generated as a series of small results, which are later combined.

/preprocessing: Holds programs for further processing of prediction results. This includes cleaning up multiple small prediction results and counting them individually.

/test: Contains all tests performed in the paper.

Additional Information
For more details regarding the code and tests, refer to the paper "Deep Learning Approach for Cell Nuclear Pore Detection and Quantification over High Resolution 3D Data", available at this link: https://hdl.handle.net/10919/117266


## Installation

To set up the required Python packages for this pipeline, please follow these steps:

1. Navigate to the project's root directory in your command-line terminal.

2. Ensure you have Python and pip installed on your system, and make sure you have an NVIDIA GPU with CUDA support. You will need to install the appropriate version of CUDA and cuDNN for your GPU.

3. Run the following command to install all the necessary packages listed in the `requirements.txt` file:

   ```bash
   pip install -r requirements.txt


## Converting N5 to Scalar N5 with Paintera Conversion Helper

This section describes how to use Paintera Conversion Helper to convert an N5 dataset into a scalar N5 dataset. It's recommended to set up a dedicated Conda environment for Paintera. This ensures that the dependencies required for the conversion do not conflict with other packages in your system.

### Step-by-Step Guide

1. **Create a New Conda Environment**: Start by creating a new environment specifically for Paintera. Replace `paintera-env` with your preferred name for the environment.
   ```bash
   conda create -n paintera-env
   ```

2. **Activate the Paintera Environment**: Before running any commands, make sure the Paintera environment is activated:
   ```bash
   conda activate paintera-env
   ```

3. **Install Paintera Conversion Helper**: Within this environment, install the Paintera Conversion Helper.
   ```bash
   conda install -c conda-forge paintera-conversion-helper
   ```

4. **Install `openjdk` and `maven`**: These are required dependencies. Install them from `conda-forge`:
   ```bash
   conda install -c conda-forge maven openjdk
   ```

5. **Run the Conversion Command**: Use the following command to convert your N5 dataset to a scalar N5 format:
   ```bash
   paintera-convert ts -i <input-path> -o <output-path> -I <label-path>
   ```
   Replace `<input-path>`, `<output-path>`, and `<label-path>` with your specific file paths and dataset name.

   Example:
   ```bash
   paintera-convert ts -i Cell1_2_Crop1_Back4.n5 -o Cell1_2_Crop1_Back4_scalar.n5 -I Labels/NuclearPore
   ```

For more information and advanced usage, refer to the [Paintera Conversion Helper documentation](https://github.com/saalfeldlab/paintera-conversion-helper).
