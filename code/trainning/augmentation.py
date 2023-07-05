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
        PadIfNeeded(image_shape,border_mode='constant', value=0, mask_value=0, p=1)
    ])

def get_augmentation_3d_grid_distortion(image_shape):
    return Compose([
        GridDistortion(p=1),
        PadIfNeeded(image_shape,border_mode='constant', value=0, mask_value=0, p=1)
    ])

def get_augmentation_3d_random_noise(image_shape):
    return Compose([
        GaussianNoise(var_limit=(0, 0.01), p=1),
        PadIfNeeded(image_shape,border_mode='constant', value=0, mask_value=0, p=1)
    ])

def get_augmentation_3d_random_rotate90(image_shape):
    return Compose([
        RandomRotate90((1, 2), p=0.5),
        PadIfNeeded(image_shape,border_mode='constant', value=0, mask_value=0, p=1)
    ])


def get_augmentation_3d_flip(image_shape):
    return Compose([
        Flip(axis=0, p=0.5),
        Flip(axis=1, p=0.5),
        Flip(axis=2, p=0.5),
        PadIfNeeded(image_shape,border_mode='constant', value=0, mask_value=0, p=1)
    ])


def get_augmentation_gfre(image_shape):
        return Compose([
        GridDistortion(p=0.2),
        Flip(axis=0, p=0.1),
        Flip(axis=1, p=0.1),
        Flip(axis=2, p=0.1),
        GaussianNoise(var_limit=(0, 0.01), p=0.2),
        ElasticTransform((0, 0.25), interpolation=2, p=0.2),
        PadIfNeeded(image_shape,border_mode='constant', value=0, mask_value=0, p=0.2)
    ])


def get_augmentation_RRF(patch_size):
    return Compose([
        RandomScale(p=0.5),
        Rotate((-15, 15), (0, 0), (0, 0), p=0.5),
        Flip(0, p=0.5),
        Flip(1, p=0.5),
        Flip(2, p=0.5),
        PadIfNeeded(patch_size,border_mode='constant', value=0, mask_value=0, p=1)
    ])


def get_augmentation_GFGE(patch_size):
    return Compose([
        GridDistortion(p=0.2),
        Flip(axis=0, p=0.1),
        Flip(axis=1, p=0.1),
        Flip(axis=2, p=0.1),
        GaussianNoise(var_limit=(0, 0.01), p=0.2),
        ElasticTransform((0, 0.25), interpolation=2, p=0.2),
        PadIfNeeded(patch_size,border_mode='constant', value=0, mask_value=0, p=1)
    ])

def get_augmentation_all(patch_size):
    return Compose([
        RandomScale(p=0.3),
        Rotate((-15, 15), (0, 0), (0, 0), p=0.3),
        GridDistortion(p=0.6),
        Flip(axis=0, p=0.2),
        Flip(axis=1, p=0.2),
        Flip(axis=2, p=0.2),
        GaussianNoise(var_limit=(0, 0.01), p=0.5),
        ElasticTransform((0, 0.25), interpolation=2, p=0.4),
        PadIfNeeded(patch_size,border_mode='constant', value=0, mask_value=0, p=1)
    ])