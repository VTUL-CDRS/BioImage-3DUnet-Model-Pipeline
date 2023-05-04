import tensorflow as tf
import keras
import segmentation_models_3D as sm
print(tf.__version__)
print(keras.__version__)
from skimage import io
from patchify import patchify, unpatchify
import numpy as np
from matplotlib import pyplot as plt
from keras import backend as K
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tifffile import imwrite
from volumentations import *
from array import array
from keras.models import load_model
import psutil

import os
import glob
import cv2

def get_augmentation(patch_size):
    return Compose([
        Rotate((-15, 15), (0, 0), (0, 0), p=0.2),
        RandomRotate90((1, 2), p=0.2),
        Flip(0, p=0.2),
        Flip(1, p=0.2),
        Flip(2, p=0.2),
    ], p=1.0)

def img_pathify(img, patch_size):
    patches = patchify(img, (patch_size, patch_size, patch_size), step=patch_size)
    result = np.reshape(patches,  (-1, patches.shape[3], patches.shape[4], patches.shape[5]))
    return result

print("here")
modelfile = '/mnt/research-data/3dunet/models/test_aug_model.h5'
inputfile1 = '/mnt/research-data/3dunet/KirchImages/image_stack.tif'
maskfile1 = '/mnt/research-data/3dunet/KirchImages/labelAllER.tif'

iterations = 5

print("here")

image = io.imread(inputfile1)

print("here")

mask = io.imread(maskfile1)

print("here")

offset = [400, 926, 2512]
shape = [241, 476, 528]
image = image[offset[0]:offset[0]+shape[0], offset[1]:offset[1]+shape[1], offset[2]:offset[2]+shape[2]]
mask = mask[offset[0]:offset[0]+shape[0], offset[1]:offset[1]+shape[1], offset[2]:offset[2]+shape[2]]

imagelist = []
imagelist.append(image)

print("here")

masklist = []
masklist.append(mask)

print("calling get_augmentation")

aug = get_augmentation((64, 512, 704))

print("here")

for i in range(4):
    data = {'image': image, 'mask': mask}
    aug_data = aug(**data)
    newimg, newlbl = aug_data['image'], aug_data['mask']
    newimg = newimg[offset[0]:offset[0]+shape[0], offset[1]:offset[1]+shape[1], offset[2]:offset[2]+shape[2]]
    imagelist.append(newimg)
    masklist.append(newlbl)

    
print("here length")

print( len(imagelist) )
print( len(masklist) )

imagepatches = []
maskpatches = []

print("here for loop")

for i in range(len(imagelist)):
    print(imagelist[i].shape, masklist[i].shape)
    img_patches = img_pathify(imagelist[i], 8)
    print("here1")
    imagepatches.append(img_patches)
    print("here2")
    mask_patches = img_pathify(masklist[i], 8)
    print("here3")
    maskpatches.append(mask_patches)
    print("here4")
    process = psutil.Process(os.getpid())
    print("Memory usage:", process.memory_info().rss)

input_img = imagepatches[0]
input_mask = maskpatches[0]
for i in range(1, 5, 1):
    input_img = np.vstack((input_img, imagepatches[i]))
    input_mask = np.vstack((input_mask, maskpatches[i]))


print("here shape")

print(input_img.shape)
print(input_mask.shape)

train_img = np.stack((input_img,)*3, axis=-1)
train_mask = np.expand_dims(input_mask, axis=4)

X_train, X_test, y_train, y_test = train_test_split(train_img, train_mask, test_size = 0.10, random_state = 0)  

X_train.shape   

# Backbones: ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'seresnet18', 
# 'seresnet34', 'seresnet50', 'seresnet101', 'seresnet152', 'seresnext50', 'seresnext101', 
# 'senet154', 'resnext50', 'resnext101', 'vgg16', 'vgg19', 'densenet121', 'densenet169', 
# 'densenet201', 'inceptionresnetv2', 'inceptionv3', 'mobilenet', 'mobilenetv2', 'efficientnetb0', 
# 'efficientnetb1', 'efficientnetb2', 'efficientnetb3', 'efficientnetb4', 'efficientnetb5', 
# 'efficientnetb6', 'efficientnetb7']
backbones = ['vgg19', 'seresnext101', 'seresnext50', 'seresnet50', 'seresnet34']

errors = []

encoder_weights = 'imagenet'
for ch in backbones:
    BACKBONE = ch #Try vgg16, efficientnetb7, inceptionv3, resnet50, or above.
    activation = 'sigmoid'
    patch_size = 64
    channels=3

    LR = 0.0001
    optim = tf.keras.optimizers.Adam(LR)

    dice_loss = sm.losses.DiceLoss(class_weights=np.array([0.25, 0.25, 0.25, 0.25])) 
    focal_loss = sm.losses.CategoricalFocalLoss()
    total_loss = dice_loss + (1 * focal_loss)
    
    # actulally total_loss can be imported directly from library, above example just show you how to manipulate with losses
    # total_loss = sm.losses.binary_focal_dice_loss # or sm.losses.categorical_focal_dice_loss 
    
    metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]
    preprocess_input = sm.get_preprocessing(BACKBONE)

    #make try catch here and log which ones are not working
    try:
        X_train_prep = preprocess_input(X_train)
        X_test_prep = preprocess_input(X_test)

            
        model = sm.Unet(BACKBONE, classes=1, 
                    input_shape=(patch_size, patch_size, patch_size, channels), 
                    encoder_weights=encoder_weights,
                    activation=activation)
    
        model.compile(optimizer = optim, loss=total_loss, metrics=metrics)
        print(model.summary())
        
        y_train = tf.cast(y_train, tf.float32)
        y_test = tf.cast(y_test, tf.float32)
        
        history=model.fit(X_train_prep, 
                y_train,
                batch_size=8, 
                epochs=iterations,
                verbose=1,
                validation_data=(X_test_prep, y_test))

        model.save(modelfile)
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(1, len(loss) + 1)
        plt.plot(epochs, loss, 'y', label='Training loss')
        plt.plot(epochs, val_loss, 'r', label='Validation loss')
        plt.title('Training and validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        nameLoss = 'loss_' + ch
        plt.savefig('/mnt/research-data/3dunet/KirchImages/Graphs/Loss_Images/' + nameLoss + '.png')
        plt.show()
        
        acc = history.history['iou_score']
        val_acc = history.history['val_iou_score']
        
        plt.plot(epochs, acc, 'y', label='Training IOU')
        plt.plot(epochs, val_acc, 'r', label='Validation IOU')
        plt.title('Training and validation IOU')
        plt.xlabel('Epochs')
        plt.ylabel('IOU')
        plt.legend()
        nameIOU = 'iou_' + ch
        plt.savefig('/mnt/research-data/3dunet/KirchImages/Graphs/IOU_Images/' + nameIOU + '.png')
        plt.show()
    except:
      errors.append(ch)