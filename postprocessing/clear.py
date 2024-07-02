from functools import partial
from pathlib import Path

import numpy as np
import numpy.typing as npt
import tifffile as tif
import tyro
from einops import rearrange
from mpire import WorkerPool
from skimage import morphology


def worker(ind: int, pred: npt.NDArray[np.uint8], threshold: int, clear_size: int):
    pred = pred[ind, ...]
    pred[pred < threshold] = 0
    pred[pred >= threshold] = 255

    labeled: npt.NDArray[int] = morphology.label(pred, connectivity=2)
    ids, cnts = np.unique(labeled, return_counts=True)

    mask = cnts < clear_size
    to_replace = np.isin(labeled, ids[mask])
    labeled[to_replace] = 0
    labeled[labeled > 0] = 255
    labeled = labeled.astype(np.uint8)

    return labeled


def main(
    pred_file: str,
    mask_file: str,
    outdir: str,
    threshold: int,
    clear_size: int,
    n_jobs: int = 10,
):
    pred = tif.imread(pred_file)
    mask = tif.imread(mask_file)

    print("Data loaded")

    pred = mask * pred
    del mask

    # Downsample for faster process
    pred = pred[::2, ::2, ::2]
    tif.imwrite(Path(outdir) / 'combine_downsampled.tif', compression='zlib')

    pred = rearrange(pred, "(n x) y z -> n x y z", x=64)

    with WorkerPool(n_jobs=n_jobs) as pool:
        func = partial(worker, pred=pred, threshold=threshold, clear_size=clear_size)
        results = pool.map(func, range(pred.shape[0]), progress_bar=True)
        img = np.concatenate(results)
        outfile = Path(outdir) / f"combine_cleared_{threshold}_{clear_size}.tif"
        tif.imwrite(outfile, img, compression="zlib")


if __name__ == "__main__":
    tyro.cli(main)
