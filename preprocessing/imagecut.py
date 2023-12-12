import os
import numpy as np
from tifffile import TiffFile, imwrite

input_folder = "/mnt/research-data/data/yale/Control_YM489919-1_8x8x8nm_4MHz/Control_aligned-tif-8bit"
output_folder = "/mnt/research-data/data/Health-Cell-cut"

Health_Cell_full_1 = {
        "output_file": "Health-Cell-full-1.tif",
        "top_left_x_pixel": 6000,
        "top_left_y_pixel": 3000,
        "width_pixel": 3000,
        "height_pixel": 2300,
        "z_start": 7290,
        "z_end": 8760
    }


image_list_cut = [
    {
        "output_file": "Health-Cell-full-2.tif",
        "top_left_x_pixel": 2400,
        "top_left_y_pixel": 0,
        "width_pixel": 1984,
        "height_pixel": 1728,
        "z_start": 2320,
        "z_end": 3664
    },
    {
        "output_file": "Health-Cell-full-3.tif",
        "top_left_x_pixel": 7970,
        "top_left_y_pixel": 1240,
        "width_pixel": 2112,
        "height_pixel": 1792,
        "z_start": 1880,
        "z_end": 3288
    },
    {
        "output_file": "Health-Cell-full-4.tif",
        "top_left_x_pixel": 8520,
        "top_left_y_pixel": 6200,
        "width_pixel": 1856,
        "height_pixel": 1792,
        "z_start": 2200,
        "z_end": 3352
    },
    {
        "output_file": "Health-Cell-full-5.tif",
        "top_left_x_pixel": 6100,
        "top_left_y_pixel": 980,
        "width_pixel": 1984,
        "height_pixel": 1600,
        "z_start": 0,
        "z_end": 832
    },
    {
        "output_file": "Health-Cell-full-6.tif",
        "top_left_x_pixel": 5850,
        "top_left_y_pixel": 3640,
        "width_pixel": 1728,
        "height_pixel": 1664,
        "z_start": 0,
        "z_end": 512
    },
    {
        "output_file": "Health-Cell-full-7.tif",
        "top_left_x_pixel": 3100,
        "top_left_y_pixel": 5450,
        "width_pixel": 2368,
        "height_pixel": 2048,
        "z_start": 1400,
        "z_end": 2552
    },
    {
        "output_file": "Health-Cell-full-8.tif",
        "top_left_x_pixel": 750,
        "top_left_y_pixel": 2030,
        "width_pixel": 2048,
        "height_pixel": 1920,
        "z_start": 1030,
        "z_end": 2374
    }
]


file_prefix = "YM489919-1_8x8x8nm."




def cut_image_muti(image_list, file_prefix):
    multiple = 64
    
    for i in image_list:
        output_file = i["output_file"]
        top_left_x_pixel = i["top_left_x_pixel"]
        top_left_y_pixel = i["top_left_y_pixel"]
        width_pixel = i["width_pixel"]
        height_pixel = i["height_pixel"]
        z_start = i["z_start"]
        z_end = i["z_end"]

        new_width = ((width_pixel - 1) // multiple + 1) * multiple
        new_height = ((height_pixel - 1) // multiple + 1) * multiple
        new_depth = ((z_end - z_start - 1) // multiple + 1) * multiple

        file_list = [os.path.join(input_folder, f"{file_prefix}{z:05d}.tif") for z in range(z_start, z_start + new_depth)]
        merged_data = None

        for file in file_list:
            with TiffFile(file) as tif:
                data = tif.asarray()
                data_cut = data[top_left_y_pixel:top_left_y_pixel+new_height, top_left_x_pixel:top_left_x_pixel+new_width]
                
                if merged_data is None:
                    merged_data = data_cut[np.newaxis, :, :]
                else:
                    merged_data = np.concatenate((merged_data, data_cut[np.newaxis, :, :]), axis=0)

        output_path = os.path.join(output_folder, output_file)
        imwrite(output_path, merged_data, metadata={'axes': 'ZYX'})
        print("finish:"+ output_file)


cut_image_muti(image_list_cut, file_prefix)