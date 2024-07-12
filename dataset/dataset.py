from functools import partial
from glob import glob
from multiprocessing import Pool

import numpy as np
import numpy.typing as npt
import tifffile as tif
from einops import rearrange
from torch.utils.data import Dataset


def rand_crop(img: npt.NDArray, label: npt.NDArray, roi: tuple[int, int, int]):
    h, w, d = label.shape
    sh, sw, sd = roi

    assert sh <= h and sw <= w and sd <= d

    start_h = np.random.randint(0, h - sh + 1)
    start_w = np.random.randint(0, w - sw + 1)
    start_d = np.random.randint(0, d - sd + 1)

    return img[
        start_h : start_h + sh, start_w : start_w + sw, start_d : start_d + sd
    ], label[start_h : start_h + sh, start_w : start_w + sw, start_d : start_d + sd]


def random_crop_by_label(
    ind: int,
    img: npt.NDArray,
    label: npt.NDArray,
    roi: tuple[int, int, int],
    num_samples: int,
    threshold: int,
):
    limgs, llabels = [], []
    for i in range(500):
        simg, slabel = rand_crop(img[ind, ...], label[ind, ...], roi)
        if np.sum(slabel) > threshold:
            limgs.append(simg)
            llabels.append(slabel)
            num_samples -= 1
        if num_samples == 0:
            break
    if len(limgs) == 0:
        return None, None

    images = rearrange(limgs, "n h w d -> n h w d")
    labels = rearrange(llabels, "n h w d -> n h w d")

    assert len(images.shape) == 4
    assert len(labels.shape) == 4
    return images, labels


def load_image(data_dir: str, num_samples: int = 32):
    img_files = glob(f'{data_dir.rstrip("/")}/images/*.tif')
    label_files = glob(f'{data_dir.rstrip("/")}/label/*.tif')

    assert len(img_files) > 0
    assert len(img_files) == len(label_files)

    img_files = sorted(img_files)
    label_files = sorted(label_files)

    imgs, labels = [], []
    for ifile, mfile in zip(img_files, label_files):
        imgs.append(tif.imread(ifile))
        labels.append(tif.imread(mfile))

    imgs = rearrange(imgs, "n x y z -> n x y z")
    labels = rearrange(labels, "n x y z -> n x y z")

    with Pool(processes=8) as pool:
        func = partial(
            random_crop_by_label,
            img=imgs,
            label=labels,
            roi=(64, 64, 64),
            num_samples=num_samples,
            threshold=100,
        )
        results = pool.map(func, range(imgs.shape[0]))
        oimgs = [t[0] for t in results if t[0] is not None]
        olabels = [t[1] for t in results if t[1] is not None]
        imgs = np.concatenate(oimgs, axis=0)
        labels = np.concatenate(olabels, axis=0)
        imgs = rearrange(imgs, "n x y z -> n 1 x y z")
        labels = rearrange(labels, "n x y z -> n 1 x y z")

    n = imgs.shape[0]
    indices = np.random.permutation(n)
    train_size = int(0.9 * n)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    return (
        imgs[train_indices, ...],
        labels[train_indices, ...],
        imgs[val_indices, ...],
        labels[val_indices, ...],
    )


class TifDataset(Dataset):
    def __init__(
        self,
        imgs: npt.NDArray,
        labels: npt.NDArray,
        transform=None,
    ):
        self.transform = transform

        self.imgs = imgs
        self.labels = labels

    def __len__(self):
        return self.imgs.shape[0]

    def __getitem__(self, idx):
        image, label = (
            self.imgs[idx] / np.float32(255.0),
            self.labels[idx].astype(np.float32),
        )
        d = {"image": image, "label": label}
        if self.transform:
            d = self.transform(d)
        return d
