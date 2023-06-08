from skimage import morphology, measure, segmentation, feature
from scipy import ndimage as ndi
import numpy as np
import tifffile as tif


def clear_image(input_path, file_name, i):

    image_name = input_path+file_name + str(i) + ".tif"

    print("process " + image_name)

    pores = tif.imread(image_name)

    # split
    pores = morphology.opening(pores, morphology.ball(1))

    # remove small objects
    pores = morphology.remove_small_objects(pores, min_size=50)
    
    # save to tif
    output_file = input_path + "clear" + file_name  + str(i) +  ".tif"
    tif.imwrite(output_file, pores)
    print("finish " + output_file)


input_path = r"/mnt/research-data/test_prediction/RandomAll_5000/Cell1_2_Crop1/"

file_name = "Cell1_2_Crop1_"
number = 44

for i in range(number + 1):
    clear_image(input_path, file_name, i)
