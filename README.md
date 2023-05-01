# BioImage-3DUnet-Model-Pipeline
a data curation pipeline for very large biomedical images, including AI-assisted human-in-the-loop segmentation of nanoscale images using 3DUnet deep learning. 

## Training
* [train.py](train.py): train a 3DUnet model. See `python train.py -h`

```
Backbones: ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'seresnet18',
 'seresnet34', 'seresnet50', 'seresnet101', 'seresnet152', 'seresnext50', 'seresnext101',
 'senet154', 'resnext50', 'resnext101', 'vgg16', 'vgg19', 'densenet121', 'densenet169',
 'densenet201', 'inceptionresnetv2', 'inceptionv3', 'mobilenet', 'mobilenetv2', 'efficientnetb0',
 'efficientnetb1', 'efficientnetb2', 'efficientnetb3', 'efficientnetb4', 'efficientnetb5',
 'efficientnetb6', 'efficientnetb7']
```

* [predict.py](predict.py): predict a 3D semantic segmentation using built 3D U-Net model. See `python predict.py -h`. The Backbone choices should be the same as `train.py`.


## Utilities
* [imagecut.py](utils/imagecut.py): cut large images into small image by given x,y,z coordinates and width, height, depth. See `python imagecut.py -h`
* [imagecombine.py](utils/imagecombine.py): combine small images into one large image. See `python imagecombine.py -h`
* [objectcount.py](utils/objectcount.py): count the number of objects in a given image. See `python objectcount.py -h`