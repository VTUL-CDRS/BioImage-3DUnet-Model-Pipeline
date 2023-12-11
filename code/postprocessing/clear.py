from skimage import morphology, measure, segmentation, feature
from scipy import ndimage as ndi
import numpy as np
import tifffile as tif
from skimage import io

def clear_image(input_path, file_name, i, clear_size):

    image_name = input_path+file_name + str(i) + ".tif"

    print("process " + image_name)

    pores = tif.imread(image_name)

    # split
    pores = morphology.opening(pores, morphology.ball(1))

    # remove small objects
    pores = morphology.remove_small_objects(pores, min_size = clear_size)
    
    # save to tif
    output_file = input_path + "clear_" + str(clear_size) + "_" + file_name  + str(i) +  ".tif"
    tif.imwrite(output_file, pores)
    print("finish " + output_file)


def combine(input_path, file_name, number, clear_size):

    id_name = input_path + "clear_" + str(clear_size) + "_" + file_name

    input_files = [id_name + f"{i}.tif" for i in range(0, number+1)]


    output_file = id_name + "conbine_" + str(clear_size) +".tif"

    print("start_conbine")

    first_tif = io.imread(input_files[0])
    for i in range(1, number+1):
        img = io.imread(input_files[i])
        first_tif = np.concatenate((first_tif, img))
    io.imsave(output_file, first_tif)



def clear_function(input_path,file_name,number,clear_size):
    print("start clear:" + file_name)
    for i in range(number + 1):
        clear_image(input_path, file_name, i , clear_size)
    print("finish clear:" + file_name)
    combine(input_path, file_name, number, clear_size)
    print("finish conbine:" + file_name)

    

input_path = r"/mnt/research-data/data/Health-Cell-cut/Health-Cell-full-1/"
file_name = "Health-Cell-full-1-"
number = 22
clear_size = 400
clear_function(input_path,file_name,number,clear_size)

input_path = r"/mnt/research-data/data/Health-Cell-cut/Health-Cell-full-2/"
file_name = "Health-Cell-full-2-"
number = 20
clear_size = 400
clear_function(input_path,file_name,number,clear_size)

input_path = r"/mnt/research-data/data/Health-Cell-cut/Health-Cell-full-3/"
file_name = "Health-Cell-full-3-"
number = 21
clear_size = 400
clear_function(input_path,file_name,number,clear_size)

input_path = r"/mnt/research-data/data/Health-Cell-cut/Health-Cell-full-4/"
file_name = "Health-Cell-full-4-"
number = 17
clear_size = 400
clear_function(input_path,file_name,number,clear_size)

input_path = r"/mnt/research-data/data/Health-Cell-cut/Health-Cell-full-5/"
file_name = "Health-Cell-full-5-"
number = 12
clear_size = 400
clear_function(input_path,file_name,number,clear_size)

input_path = r"/mnt/research-data/data/Health-Cell-cut/Health-Cell-full-6/"
file_name = "Health-Cell-full-6-"
number = 7
clear_size = 400
clear_function(input_path,file_name,number,clear_size)

input_path = r"/mnt/research-data/data/Health-Cell-cut/Health-Cell-full-7/"
file_name = "Health-Cell-full-7-"
number = 17
clear_size = 400
clear_function(input_path,file_name,number,clear_size)

input_path = r"/mnt/research-data/data/Health-Cell-cut/Health-Cell-full-8/"
file_name = "Health-Cell-full-8-"
number = 20
clear_size = 400
clear_function(input_path,file_name,number,clear_size)