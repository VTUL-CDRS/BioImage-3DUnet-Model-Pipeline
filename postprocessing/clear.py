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
    
    out, num = morphology.label(pores, return_num=True, connectivity=2)
    ids, cnts = np.unique(out, return_counts=True)

    mask = cnts < clear_size
    to_replace = np.isin(out, ids[mask])
    out[to_replace] = 0
    out[out > 0] = 255
    out = out.astype(np.uint8)

    # save to tif
    output_file = input_path + "clear_" + str(clear_size) + "_" + file_name  + str(i) +  ".tif"
    tif.imwrite(output_file, out, compression='zlib')
    print("finish " + output_file)


def combine(input_path, file_name, number, clear_size):

    id_name = input_path + "clear_" + str(clear_size) + "_" + file_name

    input_files = [id_name + f"{i}.tif" for i in range(0, number)]


    output_file = id_name + "conbine_" + str(clear_size) +".tif"

    print("start_conbine")

    first_tif = io.imread(input_files[0])
    for i in range(number):
        img = io.imread(input_files[i])
        first_tif = np.concatenate((first_tif, img))
    tif.imwrite(output_file, first_tif, compression='zlib')

def clear_function(input_path: str,number: int,clear_size:int,file_name: str=""):
    print("start clear:" + file_name)
    clear_partial = partial(clear_image, input_path=input_path, clear_size=clear_size, file_name=file_name)
    with WorkerPool(n_jobs=4) as pool:
        pool.map(clear_partial, range(number), progress_bar=True)
    print("finish clear:" + file_name)
    combine(input_path, file_name, number, clear_size)
    print("finish conbine:" + file_name)

    
if __name__ == '__main__':
    tyro.cli(clear_function)
