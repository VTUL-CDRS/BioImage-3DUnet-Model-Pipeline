import argparse
import glob
import keras
import numpy as np
import os
import segmentation_models_3D as sm
import sys
import tensorflow as tf

from datetime import datetime
from keras.models import load_model
from matplotlib import pyplot as plt
from patchify import patchify, unpatchify
from skimage import io
from sklearn.model_selection import train_test_split
from tifffile import imwrite
from volumentations import *


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
    plt.savefig(currenttime+'-loss.png')

    iou_score = history.history['iou_score']
    val_iou_score = history.history['val_iou_score']
    plt.figure()
    plt.plot(epochs, iou_score, 'y', label='Training IOU')
    plt.plot(epochs, val_iou_score, 'r', label='Validation IOU')
    plt.title('Training and validation IOU')
    plt.xlabel('Epochs')
    plt.ylabel('IOU')
    plt.legend()
    plt.savefig(currenttime+'-iou.png')
    plt.show()


def read_input_images(inputfolder):
    images = []

    for dir_path in glob.glob(inputfolder):
        for img_path in sorted(glob.glob(os.path.join(dir_path, "*.tif")), key=os.path.getmtime):
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


def prepare_data(imagelist, masklist, patch_size=64, test_size=0.1):
    imagepatches = []
    maskpatches = []
    for i in range(len(imagelist)):
        print(imagelist[i].shape, masklist[i].shape)
        img_patches = img_pathify(imagelist[i], patch_size)
        imagepatches.append(img_patches)
        mask_patches = img_pathify(masklist[i], patch_size)
        maskpatches.append(mask_patches)

    input_img = imagepatches[0]
    input_mask = maskpatches[0]
    for i in range(1, len(imagelist), 1):
        input_img = np.vstack((input_img, imagepatches[i]))
        input_mask = np.vstack((input_mask, maskpatches[i]))

    print(input_img.shape)
    print(input_mask.shape)

    train_img = np.stack((input_img,)*3, axis=-1)
    train_mask = np.expand_dims(input_mask, axis=4)

    X_train, X_test, y_train, y_test = train_test_split(
        train_img, train_mask, test_size=test_size, random_state=0)

    return X_train, X_test, y_train, y_test


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

    imagelist, masklist = generate_augmented_data(
        train_images, mask_images, num_augmentations=5)
    X_train, X_test, y_train, y_test = prepare_data(
        imagelist, masklist, patch_size=64, test_size=0.1)

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

    model = create_model(BACKBONE, classes=1, input_shape=(
        patch_size, patch_size, channels), encoder_weights=encoder_weights, activation=activation)
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
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument(
        'modelfile',
        help="Model filename",
        type=str)

    parser.add_argument(
        'inputfolder',
        help="Input folder path",
        type=str)

    parser.add_argument(
        'maskfolder',
        help="Mask folder path",
        type=str)

    parser.add_argument(
        'checkpointfolder',
        help="Checkpoint folder path",
        type=str)

    parser.add_argument(
        'iterations',
        help="Number of iterations",
        type=str)

    args = parser.parse_args()

    train(args.modelfile, args.inputfolder, args.maskfolder,
          args.checkpointfolder, args.iterations)


if __name__ == '__main__':
    print("Training...")
    main()
    print("Done.")

