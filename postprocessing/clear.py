from skimage import morphology, measure, segmentation, feature
from scipy import ndimage as ndi
import numpy as np
import tifffile as tif
from skimage import io
import tyro
from mpire import WorkerPool
from functools import partial

def clear_image(i, input_path, file_name, clear_size, threshold):

    image_name = input_path+file_name + str(i) + ".tif"

    print("process " + image_name)

    pores = tif.imread(image_name)

    pores[pores < threshold] = 0
    pores[pores >= threshold] = 255
    
    out, num = morphology.label(pores, return_num=True, connectivity=2)
    ids, cnts = np.unique(out, return_counts=True)

    mask = cnts < clear_size
    to_replace = np.isin(out, ids[mask])
    out[to_replace] = 0
    out[out > 0] = 255
    out = out.astype(np.uint8)

    # save to tif
    output_file = f'{input_path}clear_{clear_size}_{file_name}{i}_{threshold}.tif'
    tif.imwrite(output_file, out, compression='zlib')
    print("finish " + output_file)


def combine(input_path, file_name, number, clear_size, threshold):
    input_files = [f'{input_path}clear_{clear_size}_{file_name}{i}_{threshold}.tif' for i in range(0, number)]


    output_file = f'{input_path}combine_clear_{clear_size}_{file_name}{threshold}.tif'

    print("start_combine")

    first_tif = io.imread(input_files[0])
    for i in range(number):
        img = io.imread(input_files[i])
        first_tif = np.concatenate((first_tif, img))
    tif.imwrite(output_file, first_tif, compression='zlib')

def clear_function(input_path: str,number: int,clear_size:int, threshold:int, file_name: str="", ncores: int = 16):
    print("start clear:" + file_name)
    clear_partial = partial(clear_image, input_path=input_path, clear_size=clear_size, file_name=file_name, threshold=threshold)
    with WorkerPool(n_jobs=ncores) as pool:
        pool.map(clear_partial, range(number), progress_bar=True)
    print("finish clear:" + file_name)
    combine(input_path, file_name, number, clear_size, threshold)
    print("finish combine:" + file_name)

    
if __name__ == '__main__':
    tyro.cli(clear_function)
