# Preprocessing Folder

This folder contains scripts and tools for preprocessing medical image data. 
`fill_image.py`, which is used to resize a 3D TIFF image to a specified size (in this case, 64x64x64).
`imagecut.py` is used for extracting smaller TIFF files from larger ones

## Usage

To use the `fill_image.py` script, you can follow these steps:

1. Set the input and output paths, as well as the desired input size in the script:

   ```python
   input_tiff_path = "/mnt/research-data/data/Health-Cell-cut/Health-Cell-full-2.tif"
   output_tiff_path = "/mnt/research-data/data/Health-Cell-cut/Health-Cell-full-2_64.tif"
   input_size = (64, 64, 64)

2. To execute the script, open a terminal or command prompt and run the following command:
    ```bash
    python fill_image.py

To use the imagecut.py script, follow these steps:

1. Set the input folder, output folder, and file prefix in the script:

    ```python
    input_folder = "/mnt/research-data/data/yale/Control_YM489919-1_8x8x8nm_4MHz/Control_aligned-tif-8bit"
    output_folder = "/mnt/research-data/data/Health-Cell-cut"
    file_prefix = "YM489919-1_8x8x8nm."

Make sure input_folder points to the folder containing the large TIFF files, output_folder specifies where the extracted smaller files will be saved, and file_prefix defines the common prefix for the files you want to extract.

2. Create a list of dictionaries, where each dictionary specifies the parameters for extraction. For example:

    ```python
    image_list_cut = [
        {
            "output_file": "Health-Cell-full-1.tif",
            "top_left_x_pixel": 6000,
            "top_left_y_pixel": 3000,
            "width_pixel": 3000,
            "height_pixel": 2300,
            "z_start": 7290,
            "z_end": 8760
        }
    ]
Each dictionary should define the output file name and the extraction parameters.

3. To execute the script, open a terminal or command prompt and run the following command:

    ```bash
    python imagecut.py
