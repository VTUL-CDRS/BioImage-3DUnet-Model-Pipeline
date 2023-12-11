from skimage import morphology, measure, segmentation, feature
from scipy import ndimage as ndi
import numpy as np
import tifffile as tif


def count_pore(input_path, file_name):

    pores = tif.imread(input_path+file_name+".tif")

    pores = pores.astype(bool)

    print("start with " + input_path+file_name+".tif")

    print(pores.shape)

    distance_measures = {
        'euclidean': {'function': ndi.distance_transform_edt, 'min_distance': 13},
        'manhattan': {'function': lambda x: ndi.distance_transform_cdt(x, metric='taxicab'), 'min_distance': 14},
        'chebyshev': {'function': lambda x: ndi.distance_transform_cdt(x, metric='chessboard'), 'min_distance': 15}
    }

    count_number = {'euclidean': 0, 'manhattan': 0, 'chebyshev': 0}

    for name, info in distance_measures.items():

        #distance function
        distance = info['function'](pores)
        
        #find the coorrds
        coords = feature.peak_local_max(distance, min_distance=info['min_distance'], labels=pores)
        
        # mark
        mask = np.zeros(distance.shape, dtype=bool)
        mask[tuple(coords.T)] = True
        markers, _ = ndi.label(mask)
        
        # use watershed
        labels = segmentation.watershed(-distance, markers, mask=pores)
        
        # count label
        num_pores = len(np.unique(labels)) - 1 
        
        print(f'Number of pores ({name}):', num_pores)
        
        # save to tif
        output_file = input_path + fr"{file_name}_{name}_{num_pores}.tif"
        tif.imwrite(output_file, labels)
        count_number[name] = num_pores
    return count_number

def count_function(input_path,file_name,number):
    total_number = {'euclidean': 0, 'manhattan': 0, 'chebyshev': 0}
    input_files = [file_name + f"{i}" for i in range(0, number+1)]
    for name in input_files:
        current_result = count_pore(input_path, name)
        for key in current_result:
            total_number[key] += current_result[key]

    print(file_name + " Total pores:", total_number)

input_path = r"/mnt/research-data/data/Health-Cell-cut/Health-Cell-full-1/"
file_name = "clear_400_Health-Cell-full-1-"
number = 22
count_function(input_path,file_name,number)

input_path = r"/mnt/research-data/data/Health-Cell-cut/Health-Cell-full-2/"
file_name = "clear_400_Health-Cell-full-2-"
number = 20
count_function(input_path,file_name,number)

input_path = r"/mnt/research-data/data/Health-Cell-cut/Health-Cell-full-3/"
file_name = "clear_400_Health-Cell-full-3-"
number = 21
count_function(input_path,file_name,number)

input_path = r"/mnt/research-data/data/Health-Cell-cut/Health-Cell-full-4/"
file_name = "clear_400_Health-Cell-full-4-"
number = 17
count_function(input_path,file_name,number)

input_path = r"/mnt/research-data/data/Health-Cell-cut/Health-Cell-full-5/"
file_name = "clear_400_Health-Cell-full-5-"
number = 12
count_function(input_path,file_name,number)

input_path = r"/mnt/research-data/data/Health-Cell-cut/Health-Cell-full-6/"
file_name = "clear_400_Health-Cell-full-6-"
number = 7
count_function(input_path,file_name,number)

input_path = r"/mnt/research-data/data/Health-Cell-cut/Health-Cell-full-7/"
file_name = "clear_400_Health-Cell-full-7-"
number = 17
count_function(input_path,file_name,number)

input_path = r"/mnt/research-data/data/Health-Cell-cut/Health-Cell-full-8/"
file_name = "clear_400_Health-Cell-full-8-"
number = 20
count_function(input_path,file_name,number)


