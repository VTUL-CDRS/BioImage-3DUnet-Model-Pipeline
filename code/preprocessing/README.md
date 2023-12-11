# Preprocessing Folder

This folder contains scripts and tools for preprocessing medical image data. One of the key scripts in this folder is `fill_image.py`, which is used to resize a 3D TIFF image to a specified size (in this case, 64x64x64).

## Usage

To use the `fill_image.py` script, you can follow these steps:

1. Set the input and output paths, as well as the desired input size in the script:

   ```python
   input_tiff_path = "/mnt/research-data/data/Health-Cell-cut/Health-Cell-full-2.tif"
   output_tiff_path = "/mnt/research-data/data/Health-Cell-cut/Health-Cell-full-2_64.tif"
   input_size = (64, 64, 64)

2. To execute the script in bash:
    ```bash
    python fill_image.py