import SimpleITK as sitk
import numpy as np
from tifffile import TiffFile

def extend_image_to_multiple_3d(image_path, save_path, multiple):

    with TiffFile(image_path) as tif:
        image_data = tif.asarray()


    image = sitk.GetImageFromArray(image_data)
    width, height, depth = image.GetSize()


    new_width = ((width - 1) // multiple + 1) * multiple
    new_height = ((height - 1) // multiple + 1) * multiple
    new_depth = ((depth - 1) // multiple + 1) * multiple

 
    new_image = sitk.Image(new_width, new_height, new_depth, image.GetPixelID())
    new_image.SetDirection(image.GetDirection())
    new_image.SetSpacing(image.GetSpacing())
    new_image.SetOrigin(image.GetOrigin())


    paste_filter = sitk.PasteImageFilter()
    paste_filter.SetSourceSize([width, height, depth])
    paste_filter.SetSourceIndex([0, 0, 0])
    paste_filter.SetDestinationIndex([0, 0, 0])
    new_image = paste_filter.Execute(new_image, image)

    sitk.WriteImage(new_image, save_path)





input_tiff_path = "/mnt/research-data/data/Health-Cell-cut/Health-Cell-full-2.tif"
output_tiff_path = "/mnt/research-data/data/Health-Cell-cut/Health-Cell-full-2_64.tif"
multiple = 64
extend_image_to_multiple_3d(input_tiff_path, output_tiff_path, multiple)