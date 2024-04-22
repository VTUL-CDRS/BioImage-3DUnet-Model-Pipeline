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

def generate_augmented_data(train_images, mask_images, num_augmentations):
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


def plot_loss_and_iou(history, current):

    loss = history['loss']
    val_loss = history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    iou_score = history['iou_score']
    val_iou_score = history['val_iou_score']
    plt.subplot(2, 2, 2)
    plt.plot(epochs, iou_score, 'b', label='Training IOU Score')
    plt.plot(epochs, val_iou_score, 'r', label='Validation IOU Score')
    plt.title('Training and Validation IOU Score')
    plt.ylim(0, 1)
    plt.legend()

    f1_score = history['f1-score']
    val_f1_score = history['val_f1-score']
    plt.subplot(2, 2, 3)
    plt.plot(epochs, f1_score, 'b', label='Training F1 Score')
    plt.plot(epochs, val_f1_score, 'r', label='Validation F1 Score')
    plt.title('Training and Validation F1 Score')
    plt.ylim(0, 1)
    plt.legend()

    plt.tight_layout()
    plt.savefig('train_with_cyclical_augmentations_random_sample_' + current + '.png')

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
    # print("image_size: ",img.shape)
    patches = patchify(img, (patch_size, patch_size,
                       patch_size), step=patch_size)
    result = np.reshape(
        patches,  (-1, patches.shape[3], patches.shape[4], patches.shape[5]))
    return result

# def create_model(backbone, classes, input_shape, encoder_weights, activation):
#     model = sm.Unet(backbone, classes=classes, input_shape=input_shape,
#                     encoder_weights=encoder_weights, activation=activation)
#     return model
def create_model(backbone, classes, input_shape, activation):
    model = sm.Unet(backbone, classes=classes, input_shape=input_shape, activation=activation)
    return model

def compile_model(model, learning_rate, loss, metrics):
    optim = tf.keras.optimizers.Adam(learning_rate)
    model.compile(optimizer=optim, loss=loss, metrics=metrics)


def random_block_sampling(data_list, labels_list, block_size, min_positive_labels, weights, num_augmentations):

    sampled_data = None
    sampled_label = None
    while True:
        if weights is not None:
            data_index = np.random.choice(len(weights), p=np.array(weights)/sum(weights))
            ind = data_index * (num_augmentations + 1) + np.random.randint(0, num_augmentations + 1)
        else:
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


def split_data_along_longest_axis(data, train_ratio=0.8, padding_value=0):

    longest_dim = np.argmax(data.shape)

    split_idx = int(data.shape[longest_dim] * train_ratio)
    
    while (data.shape[longest_dim] - split_idx) % 64 != 0:
        split_idx -= 1

    if longest_dim == 0:
        train_data = data[:split_idx, :, :]
        test_data = data[split_idx:, :, :]
    elif longest_dim == 1:
        train_data = data[:, :split_idx, :]
        test_data = data[:, split_idx:, :]
    else:
        train_data = data[:, :, :split_idx]
        test_data = data[:, :, split_idx:]

    pad_dims = [(0, (64 - s % 64) % 64) if idx != longest_dim else (0, 0) for idx, s in enumerate(test_data.shape)]
    test_data = np.pad(test_data, pad_dims, mode='constant', constant_values=padding_value)

    return train_data, test_data

def process_all_data(images, masks, train_ratio=0.8, block_size=64):

    train_images = []
    test_images = []
    train_masks = []
    test_masks = []
    
    for img, mask in zip(images, masks):
        train_img, test_img = split_data_along_longest_axis(img, train_ratio, block_size)
        train_mask, test_mask = split_data_along_longest_axis(mask, train_ratio, block_size)
        
        train_images.append(train_img)
        test_images.append(test_img)
        train_masks.append(train_mask)
        test_masks.append(test_mask)

    return train_images, test_images, train_masks, test_masks

def compute_average(values):
    valid_values = [v for v in values if v is not None]
    return sum(valid_values) / len(valid_values)


def train(modelfile, inputfolder, maskfolder, steps_per_epoch, train_ratio, batch_size, learning_rate, cyclical_augmentations, num_augmentations, random_block_sampling_pixel, weights = None, load_model = None):

    # encoder_weights = 'imagenet'
    BACKBONE = 'vgg16'
    activation = 'sigmoid'
    block_size = (64,64,64)
    patch_size = 64
    channels = 3

    train_images = read_input_images(inputfolder)
    mask_images = read_input_images(maskfolder)

    index = 0
    for i,j in zip(train_images,mask_images):
        print(index)
        print("raw_shape: ",i.shape)
        print("mask_shape: ",j.shape)
        assert i.shape == j.shape
        index = index + 1

    if weights is not None:
        assert len(weights) == index

    dice_loss = sm.losses.DiceLoss(
        class_weights=np.array([0.25, 0.25, 0.25, 0.25]))
    focal_loss = sm.losses.CategoricalFocalLoss()
    total_loss = dice_loss + (1 * focal_loss)
    metrics = [sm.metrics.IOUScore(
        threshold=0.5), sm.metrics.FScore(threshold=0.5)]

    if load_model == "None":
        model = create_model(BACKBONE, classes=1, input_shape=
        # (patch_size, patch_size, patch_size, channels), encoder_weights=encoder_weights, activation=activation)
        (patch_size, patch_size, patch_size, channels), activation=activation)
        compile_model(model, learning_rate, total_loss, metrics)
    else:
        model = tf.keras.models.load_model(load_model, custom_objects={
            'dice_loss_plus_1focal_loss': total_loss,
            'iou_score': metrics[0],
            'f1-score': metrics[1]
            })
    print(model.summary())

    ACKBONE = 'vgg16'
    preprocess_input = sm.get_preprocessing(BACKBONE)
    X_train, X_test, y_train, y_test = process_all_data(train_images, mask_images, train_ratio)

    X_test, y_test = prepare_data(X_test, y_test, patch_size)
    X_test = preprocess_input(X_test)
    y_test = tf.cast(y_test, tf.float32)

    # record history
    history_total = {'f1-score':[], 'val_f1-score': [], 'loss': [], 'val_loss': [], 'iou_score': [], 'val_iou_score': []}
    history_period = {'f1-score':[], 'val_f1-score': [], 'loss': [], 'val_loss': [], 'iou_score': [], 'val_iou_score': []}
    
    augmentation_threshold_f = 0.05
    augmentation_threshold_iou = 0.05

    print(f"start f1 augmentation threshold:{augmentation_threshold_f}")
    print(f"start iou augmentation threshold:{augmentation_threshold_iou}")
    recovery_augmentations = False
    recovery_stop_check = False
    recovery_iou = 0
    recovery_f1 = 0
    recovery_stops = 0

    early_stop_check = False
    early_stop = 0
    early_stop_number = 7
    epoch = 1
    dynamic_cyclical_augmentations = True
    augmentations_time = 0
    check_epoch = cyclical_augmentations

    while (early_stop != early_stop_number) and (recovery_stops != early_stop_number):
        print("Epoch:",epoch)
        if dynamic_cyclical_augmentations:
            now = datetime.now()
            currenttime = now.strftime("%Y-%m-%d_%H_%M_%S")
            print("generate_augmented_data" + currenttime)
            X_train_a, y_train_a = generate_augmented_data(X_train, y_train, num_augmentations=num_augmentations)
            now = datetime.now()
            currenttime = now.strftime("%Y-%m-%d_%H_%M_%S")
            print("finish generate_augmented_data" + currenttime)
            dynamic_cyclical_augmentations = False
            augmentations_time = augmentations_time + 1
            print("create augmentation " + str(augmentations_time) + " times")

       
        history = None
        sampled_data = []
        sampled_labels = []
        now = datetime.now()
        currenttime = now.strftime("%Y-%m-%d_%H_%M_%S")
        print("random_block_sampling" + currenttime)
        for i in range(int(steps_per_epoch)):
            sampled_data_c, sampled_labels_c = random_block_sampling(X_train_a, y_train_a, block_size, random_block_sampling_pixel, weights, num_augmentations=num_augmentations)
            sampled_data.append(sampled_data_c)
            sampled_labels.append(sampled_labels_c)
        now = datetime.now()
        currenttime = now.strftime("%Y-%m-%d_%H_%M_%S")
        print("finish random_block_sampling" + currenttime)

        X_train_c, y_train_c = prepare_data(sampled_data, sampled_labels, patch_size)
        X_train_c = preprocess_input(X_train_c)
        y_train_c = tf.cast(y_train_c, tf.float32)
        history = model.fit(X_train_c, y_train_c,validation_data=(X_test, y_test), batch_size=batch_size, epochs=1,verbose=1)

        history_total['loss'].append(history.history['loss'][0])
        history_total['f1-score'].append(history.history['f1-score'][0])
        history_total['iou_score'].append(history.history['iou_score'][0])
        history_total['val_iou_score'].append(history.history['val_iou_score'][0])
        history_total['val_loss'].append(history.history['val_loss'][0])
        history_total['val_f1-score'].append(history.history['val_f1-score'][0])

        
        if (epoch) % check_epoch == 0:
            average_train_loss = compute_average(history_total['loss'][-check_epoch:])
            average_val_loss = compute_average(history_total['val_loss'][-check_epoch:])
            average_train_f1 = compute_average(history_total['f1-score'][-check_epoch:])
            average_train_iou = compute_average(history_total['iou_score'][-check_epoch:])
            average_val_f1 = compute_average(history_total['val_f1-score'][-check_epoch:])
            average_val_iou = compute_average(history_total['val_iou_score'][-check_epoch:])
            print(f"Average Training Loss over {check_epoch} epochs: {average_train_loss:.4f}")
            print(f"Average Validation Loss over {check_epoch} epochs: {average_val_loss:.4f}")
            print(f"Average Training F1 Score over {check_epoch} epoch: {average_train_f1:.4f}")
            print(f"Average Training IOU Score over {check_epoch} epoch: {average_train_iou:.4f}")
            print(f"Average Validation F1 Score over {check_epoch} epoch: {average_val_f1:.4f}")
            print(f"Average Validation IOU Score over {check_epoch} epoch: {average_val_iou:.4f}")
            history_period['loss'].append(average_train_loss)
            history_period['val_loss'].append(average_val_loss)
            history_period['f1-score'].append(average_train_f1)
            history_period['iou_score'].append(average_train_iou)
            history_period['val_f1-score'].append(average_val_f1)
            history_period['val_iou_score'].append(average_val_iou)

            if len(history_period['f1-score']) > 1:
                last_f1_score = history_period['val_f1-score'][-1]
                second_last_f1_score = history_period['val_f1-score'][-2]
                last_val_iou_score = history_period['val_iou_score'][-1]
                second_last_val_iou_score = history_period['val_iou_score'][-2]
                
                val_f1_score_increase = last_f1_score - second_last_f1_score
                val_iou_score_increase = last_val_iou_score - second_last_val_iou_score
                print(f"Validation F1 Score increase over the last {check_epoch} epochs: {val_f1_score_increase}")
                print(f"Validation IOU Score increase over the last {check_epoch} epochs: {val_iou_score_increase}")
                print(f"val_f1_score_increase {val_f1_score_increase}")
                print(f"val_iou_score_increase {val_iou_score_increase}")

                if recovery_augmentations:
                    print("Recovery augmentations active.")
                    if last_val_iou_score < recovery_iou or last_f1_score < recovery_f1:
                        print(f"Current validation IOU: {last_val_iou_score} (Target: {recovery_iou}), Current F1 score: {last_f1_score} (Target: {recovery_f1})")
                        print("Performance below threshold, continuing dynamic cyclical augmentations.")
                        dynamic_cyclical_augmentations = True
                        if recovery_stop_check:
                            recovery_stops += 1
                            print(f"Incrementing recovery stops to {recovery_stops}.")
                        else:
                            recovery_stop_check = True
                            print("Recovery stop check activated.")
                    else:
                        print("Performance targets met. Stopping recovery augmentations.")
                        recovery_augmentations = False
                        recovery_stops = 0
                        recovery_stop_check = False
                else:

                    if val_f1_score_increase < augmentation_threshold_f:
                        dynamic_cyclical_augmentations = True
                        print(f"val_f1_score_increase less than {augmentation_threshold_f}")
                        
                        if (val_f1_score_increase < 0):
                            print(f"val_f1_score_increase less than 0")
                            print(f"enter recovery augmentations")
                            recovery_iou = last_val_iou_score
                            recovery_f1 = last_f1_score
                            print(f"recover val_iou:" + str(recovery_iou))
                            print(f"recover val_f1:" + str(recovery_f1))
                            recovery_augmentations = True
                            
                        else:
                            augmentation_threshold_f = val_f1_score_increase
                            print(f"f1 augmentation threshold dacay to {augmentation_threshold_f}")

                        if val_iou_score_increase < augmentation_threshold_iou:
                            print(f"val_iou_score_increase less than {augmentation_threshold_iou}")

                            if (val_iou_score_increase < 0):
                                print(f"val_iou_score_increase less than 0")
                                print(f"enter recovery augmentations")
                                recovery_iou = last_val_iou_score
                                recovery_f1 = last_f1_score
                                print(f"recover val_iou:" + str(recovery_iou))
                                print(f"recover val_f1:" + str(recovery_f1))
                                recovery_augmentations = True
                            else:
                                augmentation_threshold_iou = val_iou_score_increase
                                print(f"augmentation threshold dacay to {augmentation_threshold_iou}")

                        if early_stop_check:
                            early_stop = early_stop + 1
                            print("early stop count:",early_stop)
                        else:
                            early_stop_check = True

                    elif val_iou_score_increase < augmentation_threshold_iou:
                        dynamic_cyclical_augmentations = True
                        if (val_iou_score_increase < 0):
                            print(f"val_iou_score_increase less than 0")
                            print(f"enter recovery augmentations")
                            recovery_iou = last_val_iou_score
                            recovery_f1 = last_f1_score
                            print(f"recover val_iou:" + str(recovery_iou))
                            print(f"recover val_f1:" + str(recovery_f1))
                            recovery_augmentations = True
                        else:
                            augmentation_threshold_iou = val_iou_score_increase
                            print(f"augmentation threshold dacay to {augmentation_threshold_iou}")

                        if early_stop_check:
                            early_stop = early_stop + 1
                            print("early stop count:",early_stop)
                        else:
                            early_stop_check = True
                        
                    else:
                        early_stop_check = False
                        early_stop = 0
                        print("not early stop point")

            model.save(modelfile + "dynamic_cyclical_augmentations" + str(epoch) + ".h5")
        
        sampled_data = []
        sampled_labels = []
        X_train_c = None
        y_train_c = None
        gc.collect()
        epoch = epoch + 1 

    print("total create augmentation " + str(augmentations_time) + " times")
    plot_loss_and_iou(history_total, "total")
    plot_loss_and_iou(history_period, "period")

    # for epoch in range(epochs):
    #     print("Epoch:",epoch+1+start_epochs)
    #     if epoch % cyclical_augmentations == 0:
    #         now = datetime.now()
    #         currenttime = now.strftime("%Y-%m-%d_%H_%M_%S")
    #         print("generate_augmented_data" + currenttime)
    #         X_train_a, y_train_a = generate_augmented_data(X_train, y_train, num_augmentations=num_augmentations)
    #         now = datetime.now()
    #         currenttime = now.strftime("%Y-%m-%d_%H_%M_%S")
    #         print("finish generate_augmented_data" + currenttime)      
          
    #     history = None
    #     sampled_data = []
    #     sampled_labels = []
    #     now = datetime.now()
    #     currenttime = now.strftime("%Y-%m-%d_%H_%M_%S")
    #     print("random_block_sampling" + currenttime)
    #     for i in range(int(steps_per_epoch)):
    #         sampled_data_c, sampled_labels_c = random_block_sampling(X_train_a, y_train_a, block_size, random_block_sampling_pixel, weights, num_augmentations=num_augmentations)
    #         sampled_data.append(sampled_data_c)
    #         sampled_labels.append(sampled_labels_c)
    #     now = datetime.now()
    #     currenttime = now.strftime("%Y-%m-%d_%H_%M_%S")
    #     print("finish random_block_sampling" + currenttime)

    #     X_train_c, y_train_c = prepare_data(sampled_data, sampled_labels, patch_size)
    #     X_train_c = preprocess_input(X_train_c)
    #     y_train_c = tf.cast(y_train_c, tf.float32)
    #     history = model.fit(X_train_c, y_train_c,validation_data=(X_test, y_test), batch_size=batch_size, epochs=1,verbose=1)
        
    #     history_total['loss'].append(history.history['loss'][0])
    #     history_total['f1-score'].append(history.history['f1-score'][0])
    #     history_total['iou_score'].append(history.history['iou_score'][0])
    #     history_total['val_iou_score'].append(history.history['val_iou_score'][0])
    #     history_total['val_loss'].append(history.history['val_loss'][0])
    #     history_total['val_f1-score'].append(history.history['val_f1-score'][0])

    #     if epoch != 0 and (epoch+1) % 50 == 0:
    #         model.save(modelfile + "random_block_sampling_v4_augmentations" + str(epoch+1+start_epochs) + ".h5")

    #     sampled_data = []
    #     sampled_labels = []
    #     X_train_c = None
    #     y_train_c = None
    #     gc.collect()

    # plot_loss_and_iou(history_total, currenttime)


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
    load_model = parameters["load_model"]
    # start_epochs = int(parameters["start_epochs"])
    # epochs = int(parameters["epochs"])
    steps_per_epoch = int(parameters["steps_per_epoch"])

    print(f"modelfile: {modelfile}")
    print(f"inputfolder: {inputfolder}")
    print(f"maskfolder: {maskfolder}")
    print(f"load_model: {load_model}")
    # print(f"start_epochs: {start_epochs}")
    # print(f"epochs: {epochs}")
    print(f"steps_per_epoch: {steps_per_epoch}")

    batch_size = 1
    learning_rate = 0.0001
    cyclical_augmentations = 25
    train_ratio = 0.8
    num_augmentations = 5
    random_block_sampling_pixel = 2000
    # weights = [1,1,1,1,1,4]
    weights = None

    train(modelfile, inputfolder, maskfolder, steps_per_epoch, train_ratio, batch_size, learning_rate, cyclical_augmentations, num_augmentations, random_block_sampling_pixel, weights = weights, load_model = load_model)

if __name__ == '__main__':
    print("Training...")
    main()
    print("Done.")

