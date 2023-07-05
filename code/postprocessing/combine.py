import numpy as np
import tifffile as tif

ind = "8"

input_files = ["/mnt/research-data/data/Health-Cell-cut/Health-Cell-full-"+ind+f"/{i}.tif" for i in range(0, 21)]


output_file = "/mnt/research-data/data/Health-Cell-cut/Health-Cell-full-"+ind+"/Health-Cell-full-"+ind+"_combine_prediction.tif"

first_tif = tif.imread(input_files[0])
metadata_tags = tif.TiffFile(input_files[0]).pages[0].tags
metadata_dict = {}
for tag in metadata_tags:
    metadata_dict[tag.name] = tag.value


shape = first_tif.shape
dtype = first_tif.dtype


tif.imwrite(output_file, np.zeros(shape, dtype=dtype), metadata=metadata_dict, bigtiff=True)


for input_file in input_files:
    tif_data = tif.imread(input_file)
    tif.imwrite(output_file, tif_data, append=True)


merged_tif = tif.imread(output_file)
merged_metadata_tags = tif.TiffFile(output_file).pages[0].tags
merged_metadata_dict = {}
for tag in merged_metadata_tags:
    merged_metadata_dict[tag.name] = tag.value
