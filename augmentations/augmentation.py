import numpy as np
from volumentations import Compose, RandomScale, Rotate
from volumentations import ElasticTransform, GridDistortion, GaussianNoise
from volumentations import RandomRotate90, Normalize, Flip ,PadIfNeeded


def get_augmentation_3d_scale_rotate(image_shape):
    return Compose([
        RandomScale(p=0.5),
        Rotate((-15, 15), (0, 0), (0, 0), p=0.5),
        PadIfNeeded(image_shape,border_mode='constant', value=0, mask_value=0, p=1)
    ])

def get_augmentation_3d_elastic_transform(image_shape):
    return Compose([
        ElasticTransform((0, 0.25), interpolation=2, p=0.1),
        PadIfNeeded(image_shape,border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0, p=1)
    ])

def get_augmentation_3d_grid_distortion(image_shape):
    return Compose([
        GridDistortion(p=1),
        PadIfNeeded(image_shape,border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0, p=1)
    ])

def get_augmentation_3d_random_noise(image_shape):
    return Compose([
        GaussianNoise(var_limit=(0, 0.01), p=1),
        PadIfNeeded(image_shape,border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0, p=1)
    ])

def get_augmentation_3d_random_rotate90(image_shape):
    return Compose([
        RandomRotate90((1, 2), p=0.5),
        PadIfNeeded(image_shape,border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0, p=1)
    ])

def get_augmentation_3d_normalize(image_shape):
    return Compose([
        Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), max_pixel_value=1.0, p=1),
        PadIfNeeded(image_shape,border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0, p=1)
    ])

def get_augmentation_3d_flip(image_shape):
    return Compose([
        Flip(axis=0, p=0.5),
        Flip(axis=1, p=0.5),
        Flip(axis=2, p=0.5),
        PadIfNeeded(image_shape,border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0, p=1)
    ])
