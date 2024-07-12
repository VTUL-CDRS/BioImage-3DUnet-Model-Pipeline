import numpy as np
import tifffile as tif
import torch
import tyro
from einops import rearrange
from patchify import patchify, unpatchify
from tqdm import tqdm

from model.net import Net


def predict(ckpt: str, inputfile: str, outputfile: str):
    model = Net.load_from_checkpoint(ckpt)
    model.eval()

    inputimage = tif.imread(inputfile)

    patches = patchify(inputimage, (64, 64, 64), step=64)

    sliced_images = []
    print(patches.shape)
    for i in range(patches.shape[0]):
        predicted_patches = []
        for j in tqdm(range(patches.shape[1])):
            for k in range(patches.shape[2]):
                single_patch = patches[i, j, k, :, :, :]
                tf_patch = torch.tensor(single_patch / np.float32(255.0), device="cuda")
                tf_patch = rearrange(tf_patch, "x y z -> 1 1 x y z")
                with torch.no_grad():
                    pred = model(tf_patch).cpu().numpy()
                pred = (pred * 255.0).astype(np.uint8)
                predicted_patches.append(pred)

        predicted_patches = np.array(predicted_patches)

        predicted_patches_reshaped = np.reshape(
            predicted_patches,
            (
                1,
                patches.shape[1],
                patches.shape[2],
                patches.shape[3],
                patches.shape[4],
                patches.shape[5],
            ),
        )

        reconstructed_image = unpatchify(
            predicted_patches_reshaped, (64, inputimage.shape[1], inputimage.shape[2])
        )

        sliced_images.append(reconstructed_image)

    predicted = np.concatenate(sliced_images, axis=0)
    tif.imwrite(outputfile, predicted, compression="zlib")


if __name__ == "__main__":
    print("Predicting...")
    tyro.cli(predict)
    print("Done.")
