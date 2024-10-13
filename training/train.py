import lightning as L
import numpy as np
import torch
import tyro
from dataset.dataset import TifDataset, load_image
from lightning.pytorch import loggers as pl_loggers
from model.net import Net
from monai.transforms import Compose, RandAffined, RandFlipd, RandGaussianNoised
from torch.utils.data import DataLoader


def setup_model(model_dir):
    model = Net()

    assert torch.cuda.device_count() == 1
    accelerator = "gpu"
    devices = 1

    checkpoint = L.pytorch.callbacks.ModelCheckpoint(
        dirpath=model_dir,
        filename="{epoch}-{val_dice_loss:.4f}",
        monitor="val_dice_loss",
        every_n_epochs=20,
        save_top_k=3,
    )
    callbacks = [checkpoint]

    logger = pl_loggers.MLFlowLogger(
        experiment_name=model_dir,
        tracking_uri="http://45.3.96.243:8088",
        synchronous=False,
    )

    trainer = L.Trainer(
        accelerator=accelerator,
        devices=devices,
        max_epochs=500,
        log_every_n_steps=1,
        strategy="auto",
        callbacks=callbacks,
        check_val_every_n_epoch=1,
        logger=logger,
    )

    return (model, trainer)


def train(
    data_dir: str,
    batch_size: int = 8,
    roi: int = 64,
    model_dir: str = "./model",
    num_samples: int = 256,
    filter: str = "",
):
    """Function

    Args:
        filter: example 3,4,5
    """
    print("Loading data...")
    transform = Compose(
        [
            RandAffined(
                keys=["image", "label"],
                prob=0.15,
                rotate_range=(
                    np.pi / 6,
                    np.pi / 6,
                    np.pi / 6,
                ),  # 3 parameters control the transform on 3 dimensions
                scale_range=(0.15, 0.15, 0.15),
                mode=("bilinear", "nearest"),
                padding_mode="border",
            ),
            RandGaussianNoised("image", prob=0.15, std=0.1),
            RandFlipd(keys=["image", "label"], prob=0.3, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.3, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=0.3, spatial_axis=2),
        ]
    )

    filter_list = []
    if len(filter) > 0:
        filter_list = [int(n) for n in filter.split(",")]
    train_imgs, train_labels, val_imgs, val_labels = load_image(
        data_dir, num_samples, roi, filter_list
    )

    train_ds = TifDataset(train_imgs, train_labels, transform)
    val_ds = TifDataset(val_imgs, val_labels)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=16
    )
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=8)

    print("Setting up model...")
    (model, trainer) = setup_model(model_dir=model_dir)

    print("Starting training...")
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)


if __name__ == "__main__":
    tyro.cli(train)
