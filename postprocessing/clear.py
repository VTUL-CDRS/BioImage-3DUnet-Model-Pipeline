from skimage import morphology, measure, segmentation, feature
from scipy import ndimage as ndi
import numpy as np
import tifffile as tif
from skimage import io
import tyro
from mpire import WorkerPool
from functools import partial

def clear_image(i, input_path, file_name, clear_size):

    image_name = input_path+file_name + str(i) + ".tif"

    print("process " + image_name)

    pores = tif.imread(image_name)

    # split
    pores = np.where(pores > 127, 255, 0).astype(np.uint8)

    # remove small objects
    pores = morphology.remove_small_objects(pores, min_size = clear_size)
    
    # save to tif
    output_file = input_path + "clear_" + str(clear_size) + "_" + file_name  + str(i) +  ".tif"
    tif.imwrite(output_file, pores, compression='zlib')
    print("finish " + output_file)


def combine(input_path, file_name, number, clear_size):

    id_name = input_path + "clear_" + str(clear_size) + "_" + file_name

    input_files = [id_name + f"{i}.tif" for i in range(0, number)]


    output_file = id_name + "combine_" + str(clear_size) +".tif"

    print("start_combine")

    first_tif = io.imread(input_files[0])
    for i in range(1, number):
        img = io.imread(input_files[i])
        first_tif = np.concatenate((first_tif, img))
    tif.imwrite(output_file, first_tif, compression='zlib')

def clear_function(input_path: str,number: int,clear_size:int,file_name: str="", ncores: int = 8):
    print("start clear:" + file_name)
    clear_partial = partial(clear_image, input_path=input_path, clear_size=clear_size, file_name=file_name)
    with WorkerPool(n_jobs=ncores) as pool:
        pool.map(clear_partial, range(number), progress_bar=True)
    print("finish clear:" + file_name)
    combine(input_path, file_name, number, clear_size)
    print("finish combine:" + file_name)

    
if __name__ == '__main__':
    tyro.cli(clear_function)
