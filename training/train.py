import lightning as L
import torch
import tyro
from monai.transforms import Compose, RandFlipd, RandRotated, RandSpatialCropd
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
        filename="{epoch}-{val_rec_loss:.4f}",
        monitor="val_dice_loss",
        every_n_epochs=1000,
        save_top_k=3,
    )
    callbacks = [checkpoint]

    trainer = L.Trainer(
        accelerator=accelerator,
        devices=devices,
        max_epochs=50000,
        log_every_n_steps=1,
        strategy="auto",
        callbacks=callbacks,
        check_val_every_n_epoch=10,
    )

    return (model, trainer)


def train(data_dir: str, batch_size: int = 4, model_dir: str = "./model"):
    print("Loading data...")
    transform = Compose(
        [
            RandSpatialCropd(
                keys=["image", "label"], roi_size=(64, 64, 64), random_size=False
            ),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            RandRotated(
                keys=["image", "label"], prob=0.2, range_x=0.4, range_y=0.4, range_z=0.4
            ),
        ]
    )

    train_ds = TifDataset(data_dir, val=False, transform=transform)
    val_ds = TifDataset(data_dir, val=True)
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
