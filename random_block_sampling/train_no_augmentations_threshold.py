import argparse
import glob
import keras
import numpy as np
import os
import segmentation_models_3D as sm
import sys
import tensorflow as tf
import glob
import re
from skimage import io

from datetime import datetime
from keras.models import load_model
from matplotlib import pyplot as plt
from patchify import patchify, unpatchify
from skimage import io
from sklearn.model_selection import train_test_split
from tifffile import imwrite
from volumentations import *
from augmentation import *


def get_augmentation(patch_size):
    return Compose([
        Rotate((-30, 30), (0, 0), (0, 0), p=0.5),
        RandomRotate90((1, 2), p=0.5),
        Flip(0, p=0.5),
        Flip(1, p=0.5),
        Flip(2, p=0.5),
    ], p=1.0)


def img_pathify(img, patch_size):
    patches = patchify(img, (patch_size, patch_size,
                       patch_size), step=patch_size)
    result = np.reshape(
        patches,  (-1, patches.shape[3], patches.shape[4], patches.shape[5]))
    return result


def plot_loss_and_iou(history, currenttime):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, 'y', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(currenttime+'threshold_augmentations_loss_test.png')

    iou_score = history.history['iou_score']
    val_iou_score = history.history['val_iou_score']
    plt.figure()
    plt.plot(epochs, iou_score, 'y', label='Training IOU')
    plt.plot(epochs, val_iou_score, 'r', label='Validation IOU')
    plt.title('Training and validation IOU')
    plt.xlabel('Epochs')
    plt.ylabel('IOU')
    plt.legend()
    plt.savefig(currenttime+'threshold_augmentations_iou_test.png')

    f1_score = history.history['f1-score']
    val_f1_score = history.history['val_f1-score']
    plt.figure()
    plt.plot(epochs, f1_score, 'y', label='f1_score')
    plt.plot(epochs, val_f1_score, 'r', label='val_f1-score')
    plt.title('Training and validation f1_score')
    plt.xlabel('Epochs')
    plt.ylabel('f1_score')
    plt.legend()
    plt.savefig(currenttime+'threshold-f1_score_augmentations_test.png')


    # plt.show()


def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [atoi(c) for c in re.split(r'(\d+)', text)]

def read_input_images(inputfolder):
    images = []

    for dir_path in glob.glob(inputfolder):
        for img_path in sorted(glob.glob(os.path.join(dir_path, "*.tif")), key=natural_keys):
            print(img_path)
            img = io.imread(img_path)
            images.append(img)

    return images


def generate_augmented_data(train_images, mask_images, num_augmentations=5):
    imagelist = []
    masklist = []

    for i in range(len(train_images)):
        img = train_images[i]
        mask = mask_images[i]
        aug = get_augmentation(img.shape)

        imagelist.append(img)
        masklist.append(mask)

        for j in range(num_augmentations):
            data = {'image': img, 'mask': mask}
            aug_data = aug(**data)
            newimg, newlbl = aug_data['image'], aug_data['mask']
            imagelist.append(newimg)
            masklist.append(newlbl)

    print("Imagelist: %d" % len(imagelist))
    print("Masklist: %d" % len(masklist))

    return imagelist, masklist


def histogram_equalization_3d(img):
    img_flat = img.flatten()

    hist, bins = np.histogram(img_flat, 256, [0,256])

    cdf = hist.cumsum()

    cdf_normalized = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())
    cdf_normalized = cdf_normalized.astype(np.uint8)

    img_eq_flat = cdf_normalized[img_flat]

    img_eq = img_eq_flat.reshape(img.shape)

    return img_eq

def binary_adaptive(img):
    binary_slices = []
    for img_2d in img:
        # Note: OpenCV expects a 8-bit, single-channel image.
        img_2d_uint8 = (img_2d / img_2d.max() * 255).astype(np.uint8)
        thresh = cv2.adaptiveThreshold(img_2d_uint8, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                    cv2.THRESH_BINARY, 11, 2)
        binary_slices.append(thresh)
    binary_adaptive = np.stack(binary_slices)
    return binary_adaptive

def prepare_data(imagelist, masklist, patch_size=64, test_size=0.1):

    adaptive_imagepatches = []
    hq_imagepatches = []
    imagepatches = []
    maskpatches = []
    for i in range(len(imagelist)):
        print(imagelist[i].shape, masklist[i].shape)

        binary_adaptive_image = binary_adaptive(imagelist[i])
        hq_image = histogram_equalization_3d(imagelist[i])

        img_patches = img_pathify(imagelist[i], patch_size)
        imagepatches.append(img_patches)
        adaptive_img_patches = img_pathify(binary_adaptive_image, patch_size)
        adaptive_imagepatches.append(adaptive_img_patches)
        hq_img_patches = img_pathify(hq_image, patch_size)
        hq_imagepatches.append(hq_img_patches)
        mask_patches = img_pathify(masklist[i], patch_size)
        maskpatches.append(mask_patches)

    input_img = imagepatches[0]
    adaptive_input_img = adaptive_imagepatches[0]
    hq_input_img = hq_imagepatches[0]
    input_mask = maskpatches[0]
    for i in range(1, len(imagelist), 1):
        input_img = np.vstack((input_img, imagepatches[i]))
        adaptive_input_img = np.vstack((adaptive_input_img, adaptive_imagepatches[i]))
        hq_input_img = np.vstack((hq_input_img, hq_imagepatches[i]))
        input_mask = np.vstack((input_mask, maskpatches[i]))

    # print(input_img.shape)
    # print(input_mask.shape)

    # train_img = np.stack((input_img,)*3, axis=-1)

    # hq_image = histogram_equalization_3d(input_img)

    # # threshold into 3 
    # mask_A = np.logical_and(input_img >= 0, input_img < 70)
    # masked_input_img_A = np.where(mask_A>0, 1, 0)

    # # For pixel values between 90 and 150 (inclusive)
    # mask_B = np.logical_and(input_img >= 70, input_img <= 180)
    # masked_input_img_B = np.where(mask_B>0, 2, 0)

    # # # For pixel values between 151 and 255 (inclusive)
    # # mask_C = np.logical_and(input_img > 90, input_img <= 255)
    # # masked_input_img_C = np.where(mask_C, input_img, 0)

    # print(train_img.shape)

    # hq_image = histogram_equalization_3d(input_img)


    # # Initialize an empty array to hold the processed images
    # processed_images = []

    # # Loop over each 3D image in the batch
    # for i in range(input_img.shape[0]):
    #     # Get the current 3D image
    #     img_3d = input_img[i, :, :, :]
        
    #     # Initialize an empty array to hold the processed slices
    #     processed_slices = []

    #     # Loop over each slice along the z-axis
    #     for j in range(img_3d.shape[0]):
    #         # Get the current slice
    #         slice_2d = img_3d[j, :, :]

    #         # Make sure it's 8-bit
    #         slice_2d_uint8 = slice_2d.astype(np.uint8)

    #         # Apply adaptive thresholding to the slice
    #         thresholded_slice = cv2.adaptiveThreshold(slice_2d_uint8, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
    #                                                 cv2.THRESH_BINARY, 11, 2)

    #         # Add the processed slice to our list
    #         processed_slices.append(thresholded_slice)

    #     # Stack the processed slices along the z-axis to form a 3D image
    #     processed_img_3d = np.stack(processed_slices, axis=0)

    #     # Add the processed 3D image to our list
    #     processed_images.append(processed_img_3d)

    # # Convert the list of processed 3D images to a numpy array
    # processed_images = np.array(processed_images)


    train_img = np.stack((input_img, adaptive_input_img, hq_input_img), axis=-1)

    train_mask = np.expand_dims(input_mask, axis=4)

    return train_img, train_mask


def create_model(backbone, classes, input_shape, encoder_weights, activation):
    model = sm.Unet(backbone, classes=classes, input_shape=input_shape,
                    encoder_weights=encoder_weights, activation=activation)
    return model


def compile_model(model, learning_rate, loss, metrics):
    optim = tf.keras.optimizers.Adam(learning_rate)
    model.compile(optimizer=optim, loss=loss, metrics=metrics)


def train_model(model, X_train, y_train, X_test, y_test, checkpoint_path, epochs, batch_size=8):
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path, save_weights_only=True, verbose=1, save_freq=200)
    model.save_weights(checkpoint_path.format(epoch=0))

    history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs,
                        verbose=1, validation_data=(X_test, y_test), callbacks=[cp_callback])
    return history


def train(modelfile, inputfolder, maskfolder, checkpointfolder, iterations):

    now = datetime.now()
    currenttime = now.strftime("%Y-%m-%d_%H_%M_%S")

    checkpoint_path = checkpointfolder + currenttime + "-{epoch:04d}.ckpt"
    iterations = int(iterations)

    train_images = read_input_images(inputfolder)
    mask_images = read_input_images(maskfolder)

    # imagelist, masklist = generate_augmented_data(
    #     train_images, mask_images, num_augmentations=5)
    for i,j in zip(train_images,mask_images):
        print("raw_shape: ",i.shape)
        print("mask_shape: ",j.shape)
        assert i.shape == j.shape


    X_train, X_test, y_train, y_test = train_test_split(
        train_images, mask_images, test_size=0.3, random_state=0)

    X_train, y_train = prepare_data(X_train, y_train, patch_size=64)

    X_test, y_test = prepare_data(X_test, y_test, patch_size=64)

    encoder_weights = 'imagenet'
    BACKBONE = 'vgg16'
    activation = 'sigmoid'
    patch_size = 64
    channels = 3

    LR = 0.0001

    dice_loss = sm.losses.DiceLoss(
        class_weights=np.array([0.25, 0.25, 0.25, 0.25]))
    focal_loss = sm.losses.CategoricalFocalLoss()
    total_loss = dice_loss + (1 * focal_loss)
    metrics = [sm.metrics.IOUScore(
        threshold=0.5), sm.metrics.FScore(threshold=0.5)]

    model = create_model(BACKBONE, classes=1, input_shape=
    (patch_size, patch_size, patch_size, channels), encoder_weights=encoder_weights, activation=activation)
    compile_model(model, LR, total_loss, metrics)
    print(model.summary())

    preprocess_input = sm.get_preprocessing(BACKBONE)
    X_train_prep = preprocess_input(X_train)
    X_test_prep = preprocess_input(X_test)
    y_train = tf.cast(y_train, tf.float32)
    y_test = tf.cast(y_test, tf.float32)

    history = train_model(model, X_train_prep, y_train,
                          X_test_prep, y_test, checkpoint_path, epochs=iterations)

    model.save(modelfile)
    plot_loss_and_iou(history, currenttime)


def main():
    # parser = argparse.ArgumentParser(
    #     description=__doc__,
    #     formatter_class=argparse.RawDescriptionHelpFormatter)

    # parser.add_argument(
    #     'modelfile',
    #     help="Model filename",
    #     type=str)

    # parser.add_argument(
    #     'inputfolder',
    #     help="Input folder path",
    #     type=str)

    # parser.add_argument(
    #     'maskfolder',
    #     help="Mask folder path",
    #     type=str)

    # parser.add_argument(
    #     'checkpointfolder',
    #     help="Checkpoint folder path",
    #     type=str)

    # parser.add_argument(
    #     'iterations',
    #     help="Number of iterations",
    #     type=str)

    # args = parser.parse_args()

    # train(args.modelfile, args.inputfolder, args.maskfolder,
    #       args.checkpointfolder, args.iterations)

    modelfile = "/home/chongyuh/3dunet/model/threshold_augmentations_test.h5"
    inputfolder = "/home/chongyuh/3dunet/data/images/"
    maskfolder = "/home/chongyuh/3dunet/data/masks/"
    checkpointfolder = "/mnt/research-data/checkpoint_test/augmentations_threshold_test/"
    epochs = 100
    print("start with")
    train(modelfile, inputfolder, maskfolder, checkpointfolder, epochs)
    print("end with")

if __name__ == '__main__':
    print("Training...")
    main()
    print("Done.")

