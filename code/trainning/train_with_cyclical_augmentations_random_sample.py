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
import gc

from datetime import datetime
from keras.models import load_model
from matplotlib import pyplot as plt
from patchify import patchify, unpatchify
from skimage import io
from sklearn.model_selection import train_test_split
from tifffile import imwrite
from volumentations import *
from augmentation import *

def generate_augmented_data(train_images, mask_images, num_augmentations=5):
    imagelist = []
    masklist = []

    for i in range(len(train_images)):
        img = train_images[i]
        mask = mask_images[i]
        aug = get_augmentation_all(img.shape)

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


def plot_loss_and_iou(history, currenttime):
    loss = history['loss']
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, 'y', label='Training loss')
    plt.title('Training loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(currenttime+'-loss.png')

    iou_score = history['iou_score']
    epochs = range(1, len(iou_score) + 1)
    plt.figure()
    plt.plot(epochs, iou_score, 'y', label='Training IOU')
    plt.title('Training IOU')
    plt.xlabel('Epochs')
    plt.ylabel('IOU')
    plt.legend()
    plt.savefig(currenttime+'-iou.png')

    val_iou_score = history['val_iou_score']
    epochs = range(1, len(val_iou_score) + 1)
    plt.figure()
    plt.plot(epochs, val_iou_score, 'y', label='Validation IOU')
    plt.title('Validation IOU')
    plt.xlabel('Epochs')
    plt.ylabel('IOU')
    plt.legend()
    plt.savefig(currenttime+'-val_iou.png')

    f1_score = history['f1-score']
    epochs = range(1, len(f1_score) + 1)
    plt.figure()
    plt.plot(epochs, f1_score, 'y', label='Training f1-score')
    plt.title('Training f1-score')
    plt.xlabel('Epochs')
    plt.ylabel('f1-score')
    plt.legend()
    plt.savefig(currenttime+'-f1-score.png')

    val_f1_score = history['val_f1-score']
    epochs = range(1, len(val_f1_score) + 1)
    plt.figure()
    plt.plot(epochs, val_f1_score, 'y', label='Validation f1-score')
    plt.title('Validation f1-score')
    plt.xlabel('Epochs')
    plt.ylabel('f1-score')
    plt.legend()
    plt.savefig(currenttime+'-Validation-f1-score.png')

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
        start_x = np.random.randint(0, max_x + 1)
        start_y = np.random.randint(0, max_y + 1)
        start_z = np.random.randint(0, max_z + 1)

        # extract data
        sampled_data = data[start_x:start_x + block_size[0], start_y:start_y + block_size[1], start_z:start_z + block_size[2]]
        sampled_label = label[start_x:start_x + block_size[0], start_y:start_y + block_size[1], start_z:start_z + block_size[2]]

        if np.sum(sampled_label) >= min_positive_labels:
            break

    return sampled_data, sampled_label

def prepare_data(imagelist, masklist, patch_size):
    imagepatches = []
    maskpatches = []
    for i in range(len(imagelist)):
        img_patches = img_pathify(imagelist[i], patch_size)
        imagepatches.append(img_patches)
        mask_patches = img_pathify(masklist[i], patch_size)
        maskpatches.append(mask_patches)

    input_img = imagepatches[0]
    input_mask = maskpatches[0]
    for i in range(1, len(imagelist), 1):
        input_img = np.vstack((input_img, imagepatches[i]))
        input_mask = np.vstack((input_mask, maskpatches[i]))

    train_img = np.stack((input_img,)*3, axis=-1)
    train_mask = np.expand_dims(input_mask, axis=4)

    return train_img, train_mask

def train(modelfile, inputfolder, maskfolder, checkpointfolder, epochs, steps_per_epoch, start_epochs, load_model = None):

    encoder_weights = 'imagenet'
    BACKBONE = 'vgg16'
    activation = 'sigmoid'
    block_size = (64,64,64)
    patch_size = 64
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

    dice_loss = sm.losses.DiceLoss(
        class_weights=np.array([0.25, 0.25, 0.25, 0.25]))
    focal_loss = sm.losses.CategoricalFocalLoss()
    total_loss = dice_loss + (1 * focal_loss)
    metrics = [sm.metrics.IOUScore(
        threshold=0.5), sm.metrics.FScore(threshold=0.5)]

    if load_model == None:
        model = create_model(BACKBONE, classes=1, input_shape=
        (patch_size, patch_size, patch_size, channels), encoder_weights=encoder_weights, activation=activation)
        compile_model(model, LR, total_loss, metrics)
    else:
        model = tf.keras.models.load_model(load_model, custom_objects={
            'dice_loss_plus_1focal_loss': total_loss,
            'iou_score': metrics[0],
            'f1-score': metrics[1]
            })
    print(model.summary())

    ACKBONE = 'vgg16'
    preprocess_input = sm.get_preprocessing(BACKBONE)
    X_train, X_test, y_train, y_test = train_test_split(
        train_images, mask_images, test_size=0.1, random_state=0)


    X_test, y_test = prepare_data(X_test, y_test, patch_size)
    X_test = preprocess_input(X_test)
    y_test = tf.cast(y_test, tf.float32)

    X_train_a, y_train_a = generate_augmented_data(X_train, y_train, num_augmentations=5)
    now = datetime.now()
    currenttime = now.strftime("%Y-%m-%d_%H_%M_%S")

    # record history
    history_total = {'f1-score':[], 'val_f1-score': [], 'loss': [], 'val_loss': [], 'iou_score': [], 'val_iou_score': []}

    for epoch in range(epochs):
        print("Epoch:",epoch+1+start_epochs)
        
        history = None
        sampled_data = []
        sampled_labels = []
        for i in range(int(steps_per_epoch)):
            sampled_data_c, sampled_labels_c = random_block_sampling(X_train_a, y_train_a, block_size, 2000)
            sampled_data.append(sampled_data_c)
            sampled_labels.append(sampled_labels_c)

        X_train_c, y_train_c = prepare_data(sampled_data, sampled_labels, patch_size)
        X_train_c = preprocess_input(X_train_c)
        y_train_c = tf.cast(y_train_c, tf.float32)
        history = model.fit(X_train_c, y_train_c,validation_data=(X_test, y_test), batch_size=1, epochs=1,verbose=1)
        
        history_total['loss'].append(history.history['loss'][0])
        history_total['f1-score'].append(history.history['f1-score'][0])
        history_total['iou_score'].append(history.history['iou_score'][0])
        history_total['val_iou_score'].append(history.history['val_iou_score'][0])
        history_total['val_loss'].append(history.history['val_loss'][0])
        history_total['val_f1-score'].append(history.history['val_f1-score'][0])

        if epoch != 0 and (epoch+1) % 25 == 0:
            X_train_a, y_train_a = generate_augmented_data(X_train, y_train, num_augmentations=5)

        if epoch != 0 and (epoch+1) % 50 == 0:
            model.save(modelfile + "random_block_sampling_v4_augmentations" + str(epoch+1+start_epochs) + ".h5")

        sampled_data = []
        sampled_labels = []
        X_train_c = None
        y_train_c = None
        gc.collect()

    plot_loss_and_iou(history_total, currenttime)


def main():

    filename = "trainning_config.txt"
    parameters = {}
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                key, value = line.split("=")
                parameters[key.strip()] = value.strip().strip('\"')

    modelfile = parameters["modelfile"]
    inputfolder = parameters["inputfolder"]
    maskfolder = parameters["maskfolder"]
    checkpointfolder = parameters["checkpointfolder"]
    load_model = parameters["load_model"]
    start_epochs = int(parameters["start_epochs"])
    epochs = int(parameters["epochs"])
    steps_per_epoch = int(parameters["steps_per_epoch"])

    print(f"modelfile: {modelfile}")
    print(f"inputfolder: {inputfolder}")
    print(f"maskfolder: {maskfolder}")
    print(f"checkpointfolder: {checkpointfolder}")
    print(f"load_model: {load_model}")
    print(f"start_epochs: {start_epochs}")
    print(f"epochs: {epochs}")
    print(f"steps_per_epoch: {steps_per_epoch}")

    train(modelfile, inputfolder, maskfolder, checkpointfolder, epochs, steps_per_epoch, start_epochs , load_model = load_model)

if __name__ == '__main__':
    print("Training...")
    main()
    print("Done.")
