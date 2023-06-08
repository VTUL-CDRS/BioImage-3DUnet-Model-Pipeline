from skimage import morphology, measure, segmentation, feature
from scipy import ndimage as ndi
import numpy as np
import tifffile as tif


def count_pore(input_path, file_name):
    # Use memmap to handle large data
    with tif.TiffFile(input_path+file_name+".tif") as tif_file:
        image = tif_file.asarray(out='memmap')
    
    pores = image

    # split
    pores = morphology.opening(pores, morphology.ball(1))

    pores = pores > 0

    # remove small objects
    pores = measure.label(pores)

    # remove small objects
    pores = morphology.remove_small_objects(pores, min_size=50)

    distance_measures = {
        'euclidean': {'function': ndi.distance_transform_edt, 'min_distance': 13},
        'manhattan': {'function': lambda x: ndi.distance_transform_cdt(x, metric='taxicab'), 'min_distance': 14},
        'chebyshev': {'function': lambda x: ndi.distance_transform_cdt(x, metric='chessboard'), 'min_distance': 15}
    }

    for name, info in distance_measures.items():

        # distance function
        distance = info['function'](pores)
        
        # find the coorrds
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
        output_file = input_path + fr"test{file_name}_{name}_{num_pores}.tif"
        tif.imsave(output_file, labels)

        labels = []


input_path = r"/mnt/research-data/test_prediction/RandomAll_5000/Cell1_2_Crop1/"

file_names = ["Cell1_2_Crop1_first_quarter",
            "Cell1_2_Crop1_second_quarter",
            "Cell1_2_Crop1_third_quarter",
            "Cell1_2_Crop1_fourth_quarter"]

for name in file_names:
    count_pore(input_path, name)
