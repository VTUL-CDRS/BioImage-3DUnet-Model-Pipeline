import argparse
import glob
import numpy as np
import os
import segmentation_models_3D as sm
import sys
import tensorflow as tf
from pathlib import Path

from keras.models import load_model
from patchify import patchify, unpatchify
from skimage import io
from tifffile import imwrite


def convert(img, target_type_min, target_type_max, target_type):
    imin = img.min()
    imax = img.max()

    a = (target_type_max - target_type_min) / (imax - imin)
    b = target_type_max - a * imax
    new_img = (a * img + b).astype(target_type)
    return new_img


def combine(img_folder, term, outputfile):
    imgs = []
    for dir_path in glob.glob(img_folder):
        for img_path in sorted(glob.glob(os.path.join(dir_path, term)), key=os.path.getmtime):
            if 'clear' in img_path:
                continue
            imgs.append(img_path)
    
    print(imgs)
    allimgs = io.imread(imgs[0])
    for i in range(1, len(imgs), 1):
        allimgs = np.vstack((allimgs, io.imread(imgs[i])))

    print(allimgs.shape)
    imwrite(outputfile, allimgs, compression='zlib')


def predict(arguments):

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument(
        'modelfile',
        help="Model filename",
        type=str)

    parser.add_argument(
        'inputfile',
        help="Input filename",
        type=str)

    parser.add_argument(
        'outputfile',
        help="Output filename",
        type=str)

    args = parser.parse_args(arguments)

    modelfile = args.modelfile
    inputfile = args.inputfile
    outputfile = args.outputfile

    model = load_model(modelfile, compile=False)
    inputimage = io.imread(inputfile)

    patches = patchify(inputimage, (64, 64, 64), step=64)
    BACKBONE = 'vgg16'  # Try vgg16, efficientnetb7, inceptionv3, resnet50
    preprocess_input = sm.get_preprocessing(BACKBONE)

    predicted_patches = []
    print(patches.shape)
    for i in range(patches.shape[0]):
        for j in range(patches.shape[1]):
            for k in range(patches.shape[2]):
                single_patch = patches[i, j, k, :, :, :]
                single_patch_3ch = np.stack((single_patch,)*3, axis=-1)
                single_patch_3ch_input = preprocess_input(
                    np.expand_dims(single_patch_3ch, axis=0))
                single_patch_prediction = model.predict(single_patch_3ch_input)
                single_patch_prediction_result = np.squeeze(
                    single_patch_prediction)
                single_patch_prediction_result = single_patch_prediction_result.astype(
                    np.float16)
                predicted_patches.append(single_patch_prediction_result)

        predicted_patches = np.array(predicted_patches)

        predicted_patches_reshaped = np.reshape(predicted_patches,
                                                (1, patches.shape[1], patches.shape[2],
                                                 patches.shape[3], patches.shape[4], patches.shape[5]))

        reconstructed_image = unpatchify(
            predicted_patches_reshaped, (64, inputimage.shape[1], inputimage.shape[2]))

        imgu8 = convert(reconstructed_image, 0, 255, np.uint8)
        imwrite(Path(outputfile) / (str(i) + '.tif'), imgu8, compression='zlib')
        predicted_patches = []

    combine(os.path.split(outputfile)[0]+"/", '*.tif', outputfile + '.tif')


if __name__ == '__main__':
    print("Predicting...")
    predict(sys.argv[1:])
    print("Done.")
