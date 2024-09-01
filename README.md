# BioImage 3DUnet Model Pipeline

BioImage 3DUnet Model Pipeline is a data curation pipeline designed for processing very large biomedical images. It includes AI-assisted human-in-the-loop segmentation of nanoscale images using 3DUnet deep learning.

## Contents
Our framework has three stages: training, prediction and postprocessing.

- **/training**: Includes the main training programs. The files here are used to train models using specified raw files and their labels.
- **/prediction**: Contains programs for making predictions on raw files using specified models. 
- **/postprocessing**: Holds programs for further processing of prediction results. The postprocessing includes clearing small objects, analysis and ploting figures.

## Additional Information
For more details regarding the code and tests, refer to the paper "Deep Learning Approach for Cell Nuclear Pore Detection and Quantification over High Resolution 3D Data", available at this link: https://hdl.handle.net/10919/117266

## Installation

To set up the required Python packages for this pipeline, please follow these steps:

1. Create conda env:
```bash
conda create --name cdrs -y python=3.8
conda activate cdrs
python -m pip install --upgrade pip
```

2. Install cudatookit 
```bash
conda install -c "nvidia/label/cuda-11.7.1" cuda-toolkit
```

3. Install other dependencies:

   ```bash
   pip install -r requirements.txt
   ```
## Data

To use this framework, you need two types of data. First is training data. Second is raw images. Training data is made of cropped raw images and its labels. Training data is cropped to 128x128x128, so its total size is smaller. On ARC, its location is:

```bash
/projects/yinlin_chen/linhan/bio/train/mix_method1
```

 The raw images is cropped to contain the whole neuron so its size is much bigger. We also provide a mask for each raw image to indicate the locations of neuron. On ARC, its location is:

 ```bash
 /projects/yinlin_chen/linhan/bio/raw_images
 ```

## Run scripts

Please enter train, prediction and postprocessing for details. Sometimes, you need to set PYTHONPATH to the root directory to enable the code to work. More elegant design is on the way.
