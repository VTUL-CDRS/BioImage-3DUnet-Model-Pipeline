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


def plot_loss_and_iou(history, currenttime, augmentation):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, 'y', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(augmentation +'-loss.png')

    iou_score = history.history['iou_score']
    val_iou_score = history.history['val_iou_score']
    plt.figure()
    plt.plot(epochs, iou_score, 'y', label='Training IOU')
    plt.plot(epochs, val_iou_score, 'r', label='Validation IOU')
    plt.title('Training and validation IOU')
    plt.xlabel('Epochs')
    plt.ylabel('IOU')
    plt.legend()
    plt.savefig(augmentation +'-iou.png')


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

def img_pathify(img, patch_size):
    patches = patchify(img, (patch_size, patch_size,
                       patch_size), step=patch_size)
    result = np.reshape(
        patches,  (-1, patches.shape[3], patches.shape[4], patches.shape[5]))
    return result

def create_model(backbone, classes, input_shape, encoder_weights, activation):
    model = sm.Unet(backbone, classes=classes, input_shape=input_shape,
                    encoder_weights=encoder_weights, activation=activation)
    return model


def compile_model(model, learning_rate, loss, metrics):
    optim = tf.keras.optimizers.Adam(learning_rate)
    model.compile(optimizer=optim, loss=loss, metrics=metrics)


def random_block_sampling(data_list, labels_list, block_size, min_positive_labels):

    # pack = 10
    # count = 0
    # data_cut = None
    # labels_cut = None
    sampled_data = None
    sampled_label = None
    while True:
        ind = np.random.randint(0, len(data_list))

        data = data_list[ind]
        label = labels_list[ind]

        x_shape = data.shape
        y_shape = label.shape

        assert x_shape == y_shape

        # range
        max_x = x_shape[0] - block_size[0]
        max_y = x_shape[1] - block_size[1]
        max_z = x_shape[2] - block_size[2]

        # start
        start_x = np.random.randint(0, max_x)
        start_y = np.random.randint(0, max_y)
        start_z = np.random.randint(0, max_z)

        # extract data
        sampled_data = data[start_x:start_x + block_size[0], start_y:start_y + block_size[1], start_z:start_z + block_size[2]]
        sampled_label = label[start_x:start_x + block_size[0], start_y:start_y + block_size[1], start_z:start_z + block_size[2]]

        if np.sum(sampled_label) >= min_positive_labels:
            sampled_data = np.expand_dims(sampled_data, axis=0)
            sampled_label = np.expand_dims(sampled_label, axis=0)
            sampled_data = np.stack((sampled_data,)*3, axis=-1)
            sampled_label = np.expand_dims(sampled_label, axis=4)
            break
            # if count == 0:
            #     data_cut = sampled_data
            # else:
            #     data_cut = np.concatenate((data_cut, sampled_data), axis=0)
            # if count == 0:
            #     labels_cut = sampled_label
            # else:
            #     labels_cut = np.concatenate((labels_cut, sampled_label), axis=0)
            # count = count + 1
        # else:
        #     print("reject 1")
    BACKBONE = 'vgg16'
    preprocess_input = sm.get_preprocessing(BACKBONE)
    sampled_data = preprocess_input(sampled_data)
    sampled_label = tf.cast(sampled_label, tf.float32)

    return sampled_data, sampled_label

def prepare_data(imagelist, masklist, patch_size=64):
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

    return train_img, train_mask

def train(modelfile, inputfolder, maskfolder, checkpointfolder, epochs, steps_per_epoch):

    encoder_weights = 'imagenet'
    BACKBONE = 'vgg16'
    activation = 'sigmoid'
    block_size = (32,32,32)
    patch_size = 32
    channels = 3
    LR = 0.0001

    train_images = read_input_images(inputfolder)
    mask_images = read_input_images(maskfolder)

    index = 0
    for i,j in zip(train_images,mask_images):
        print(index)
        print("raw_shape: ",i.shape)
        print("mask_shape: ",j.shape)
        assert i.shape == j.shape
        index = index + 1
    
    # train_images = np.array(train_images)
    # mask_images = np.array(mask_images)

    X_train, X_test, y_train, y_test = train_test_split(
        train_images, mask_images, test_size=0.3, random_state=0)



    # X_train, y_train = prepare_data(
    #     X_train, y_train, patch_size)


    # X_test, y_test = prepare_data(
    #     X_test, y_test, patch_size)

    # preprocess_input = sm.get_preprocessing(BACKBONE)
    # X_test = preprocess_input(X_test)
    # y_test = tf.cast(y_test, tf.float32)

    now = datetime.now()
    currenttime = now.strftime("%Y-%m-%d_%H_%M_%S")

    # record history
    history_total = {'loss': [], 'val_loss': [], 'iou_score': [], 'val_iou_score': []}

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam()

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

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
            
        sampled_data, sampled_labels = random_block_sampling(X_train, y_train, block_size, 50)
        val_sampled_data, val_sampled_labels = random_block_sampling(X_test, y_test, block_size,50)
        # traning
        # loss, iou_score, f_score = train_step(sampled_data, sampled_labels, loss_object, optimizer, metrics)
        cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpointfolder, save_weights_only=True, verbose=1, save_freq=200)
        model.save_weights(checkpointfolder.format(epoch=0))
        history = model.fit(sampled_data, sampled_labels, batch_size=1, epochs=1,
                verbose=1, validation_data=(val_sampled_data, val_sampled_labels), callbacks=[cp_callback])
        history_total['loss'].append(history.history['loss'][0])
        history_total['val_iou_score'].append(history.history['val_iou_score'][0])
        history_total['val_loss'].append(history.history['val_loss'][0])
        history_total['iou_score'].append(history.history['iou_score'][0])


    model.save(modelfile)

    plot_loss_and_iou(history_total, currenttime, augmentation)


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

    modelfile = "/home/chongyuh/3dunet/model/random_block_sampling.h5"
    inputfolder = "/home/chongyuh/3dunet/data/images/"
    maskfolder = "/home/chongyuh/3dunet/data/masks/"
    checkpointfolder = "/mnt/research-data/checkpoint_test/random_block_sampling/"
    epochs = 100000
    steps_per_epoch = 20
    print("start with")
    train(modelfile, inputfolder, maskfolder, checkpointfolder, epochs, steps_per_epoch)
    print("end with")

    train(args.modelfile, args.inputfolder, args.maskfolder,
          args.checkpointfolder, args.iterations)


if __name__ == '__main__':
    print("Training...")
    main()
    print("Done.")

