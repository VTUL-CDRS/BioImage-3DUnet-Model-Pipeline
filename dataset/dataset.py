from glob import glob

import numpy as np
import tifffile as tif
import torch
from einops import rearrange
from torch.utils.data import Dataset


class TifDataset(Dataset):
    def __init__(self, data_dir: str, transform=None, val: bool = False):
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

        self.imgs = rearrange(imgs, "n x y z -> n 1 x y z")
        self.labels = rearrange(labels, "n x y z -> n 1 x y z")

        if val:
            self.imgs = self.imgs[-1:, ...]
            self.labels = self.labels[-1:, ...]
        else:
            self.imgs = self.imgs[:-1, ...]
            self.labels = self.labels[:-1, ...]

        self.imgs = self.imgs / np.float32(255.0)
        self.labels = self.labels.astype(np.float32)

    def __len__(self):
        return self.imgs.shape[0]

    def __getitem__(self, idx):
        image, label = self.imgs[idx], self.labels[idx]
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

    from monai.transforms import (
        Compose,
        RandCropByPosNegLabeld,
        RandFlipd,
        RandRotated,
        ToTensord,
    )

    transform = Compose(
        [
            ToTensord(keys=["image", "label"], device="cuda"),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(64, 64, 64),
                pos=1.0,
                neg=0.2,
                num_samples=4,
            ),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            RandRotated(
                keys=["image", "label"], prob=0.2, range_x=0.4, range_y=0.4, range_z=0.4
            ),
        ]
    )

    dataset = TifDataset(
        "/home/linhan/data/FIB-SEM/control_method1/", val=True, transform=transform
    )
    for i in range(6):
        d = dataset[0]
        print(d[0].keys())
        image, label = d[0]["image"], d[0]["label"]
        print(image.shape, label.shape)
        print(torch.min(image), torch.max(image))
        print(torch.min(label), torch.max(label))
