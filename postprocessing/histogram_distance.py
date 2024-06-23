import matplotlib.pyplot as plt
import numpy as np
import tifffile as tif
import tyro
from einops import rearrange
from skimage import measure
from sklearn.neighbors import KDTree


def main(file: str, head: str):
    img = tif.imread(file)
    img = rearrange(img, "(n x) y z -> n x y z", n=8)
    points = []
    for i in range(8):
        out = measure.label(img[i])
        props = measure.regionprops(out)
        centroids = [r.centroid for r in props]
        arr = np.array(centroids)
        arr[:, 0] += i * img.shape[1]
        points.append(arr)

    pcds = np.concatenate(points, axis=0)

    kdtree = KDTree(2.0 * pcds)

    # Query the nearest neighbor for each point
    # k=2 because the nearest neighbor of a point includes the point itself, so we need the second nearest
    distances, _ = kdtree.query(2.0 * pcds, k=2)

    # The nearest neighbor distances are in the second column (index 1)
    nearest_neighbor_distances = distances[:, 1]

    plt.figure()
    plt.hist(nearest_neighbor_distances, bins=200)
    plt.title(f"Histogram {head}")
    plt.savefig(f"hist_{head}.png")


if __name__ == "__main__":
    tyro.cli(main)
