# BioImage-3DUnet-Model-Pipeline
a data curation pipeline for very large biomedical images, including AI-assisted human-in-the-loop segmentation of nanoscale images using 3DUnet deep learning. 

## Utilities
* [imagecut.py](utils/imagecut.py): cut large images into small image by given x,y,z coordinates and width, height, depth. See `python imagecut.py -h`
* [imagecombine.py](utils/imagecombine.py): combine small images into one large image. See `python imagecombine.py -h`