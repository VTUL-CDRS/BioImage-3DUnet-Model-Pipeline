from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tifffile as tif
import tyro
from einops import rearrange
from mpire import WorkerPool
from skimage import morphology


def worker(i, img, threshold):
    img = img[i, ...]
    img[img < threshold] = 0
    img[img >= threshold] = 255

    out, num = morphology.label(img, return_num=True, connectivity=2)
    print("Total num: ", num)

    ids, cnts = np.unique(out, return_counts=True)
    return cnts[1:]


def main(
    img_file: str, mask_file: str, threshold: int, n_jobs: int = 10
):
    img = tif.imread(img_file)
    mask = tif.imread(mask_file)

    img = img * mask
    del mask

    img = rearrange(img, "(b x) y z -> b x y z", x=64)
    print(img[0, ...].shape)

    with WorkerPool(n_jobs=n_jobs) as pool:
        func = partial(worker, img=img, threshold=threshold)
        results = pool.map(func, range(img.shape[0]), progress_bar=True)

        print(results[0].shape)
        cnts = rearrange(results, "b -> b")
        print(cnts.shape)

        plt.figure()
        plt.hist(cnts, bins=100)
        plt.title(f"Numbers of Pixels threshold_{threshold}")
        plt.savefig(f"hist_{threshold}.png")


if __name__ == "__main__":
    tyro.cli(main)
