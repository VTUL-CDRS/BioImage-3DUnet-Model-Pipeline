import lightning as L
import torch
import tyro
from monai.transforms import Compose, RandAffined, RandFlipd, RandGaussianNoised
from torch.utils.data import DataLoader

from dataset.dataset import TifDataset
from model.net import Net


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

    trainer = L.Trainer(
        accelerator=accelerator,
        devices=devices,
        max_epochs=10000,
        log_every_n_steps=5,
        strategy="auto",
        callbacks=callbacks,
        check_val_every_n_epoch=5,
    )

    return (model, trainer)


def train(
    data_dir: str,
    batch_size: int = 8,
    model_dir: str = "./model",
    num_samples: int = 64,
):
    print("Loading data...")
    transform = Compose(
        [
            RandAffined(
                keys=["image", "label"],
                prob=0.15,
                rotate_range=(
                    0.05,
                    0.05,
                    0.05,
                ),  # 3 parameters control the transform on 3 dimensions
                scale_range=(0.1, 0.1, 0.1),
                mode=("bilinear", "nearest"),
            ),
            RandGaussianNoised("image", prob=0.15, std=0.01),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        ]
    )

    train_ds = TifDataset(
        data_dir, val=False, transform=transform, num_samples=num_samples
    )
    val_ds = TifDataset(data_dir, val=True, num_samples=num_samples)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=16
    )
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=1)

    print("Setting up model...")
    (model, trainer) = setup_model(model_dir=model_dir)

    print("Starting training...")
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)


if __name__ == "__main__":
    tyro.cli(train)
