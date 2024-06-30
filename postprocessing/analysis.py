from functools import partial
from pathlib import Path

import numpy as np
import numpy.typing as npt
import tifffile as tif
import tyro
from einops import rearrange
from mpire import WorkerPool
from skimage import measure, morphology
from sklearn.neighbors import KDTree


def worker(ind: int, pred: npt.NDArray[np.uint8], threshold: int, clear_size: int):
    pred = pred[ind, ...]
    pred[pred < threshold] = 0
    pred[pred >= threshold] = 255

    labeled: npt.NDArray[int] = morphology.label(pred, connectivity=2)

    # Clear small objects
    ids, cnts = np.unique(labeled, return_counts=True)

    mask = cnts < clear_size
    to_replace = np.isin(labeled, ids[mask])
    labeled[to_replace] = 0

    # Find centroids
    props = measure.regionprops(labeled)
    if len(props) == 0:
        centroids = None
    else:
        centroids = np.array([cc.centroid for cc in props])
        centroids[:, 0] += 64 * ind

    return cnts[1:], centroids


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

    pred = rearrange(pred, "(n x) y z -> n x y z", x=64)

    with WorkerPool(n_jobs=n_jobs) as pool:
        func = partial(worker, pred=pred, threshold=threshold, clear_size=clear_size)
        results = pool.map(func, range(pred.shape[0]), progress_bar=True)

        cnts = np.concatenate([item[0] for item in results], axis=0)
        centroids = np.concatenate([item[1] for item in results if item[1] is not None], axis=0)

        kdtree = KDTree(centroids)
        distances, _ = kdtree.query(centroids, k=2)

        nn_distances = distances[:, 1]

        npz_file = Path(outdir) / f"stat_{clear_size}_{threshold}.npz"
        np.savez(npz_file, cnts=cnts, nn_distances=nn_distances)


if __name__ == "__main__":
    tyro.cli(main)
