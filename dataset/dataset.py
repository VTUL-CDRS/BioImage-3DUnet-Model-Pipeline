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


class TifDataset(Dataset):
    def __init__(
        self, data_dir: str, transform=None, val: bool = False, num_samples=32
    ):
        self.transform = transform

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
            self.imgs = np.concatenate(oimgs, axis=0)
            self.labels = np.concatenate(olabels, axis=0)
            self.imgs = rearrange(self.imgs, "n x y z -> n 1 x y z")
            self.labels = rearrange(self.labels, "n x y z -> n 1 x y z")

        n = self.imgs.shape[0]
        if val:
            indices = np.random.choice(n, int(0.1 * n))
        else:
            indices = np.random.choice(n, int(0.9 * n))
        self.imgs = self.imgs[indices, ...]
        self.labels = self.labels[indices, ...]

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


if __name__ == "__main__":
    dataset = TifDataset("/home/linhan/data/FIB-SEM/control_method1/", val=True)
    print(len(dataset))
    d = dataset[0]
    print(d.keys())
    image, label = d["image"], d["label"]
    print(np.min(image), np.max(image))
    print(np.min(label), np.max(label))

    dataset = TifDataset("/home/linhan/data/FIB-SEM/control_method1/", val=False)
    print(len(dataset))
    d = dataset[0]
    print(d.keys())
    image, label = d["image"], d["label"]
    print(np.min(image), np.max(image))
    print(np.min(label), np.max(label))
