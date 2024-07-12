import lightning as L
import torch
from monai.inferers import sliding_window_inference
from monai.losses import DiceLoss, FocalLoss
from monai.metrics import MeanIoU
from monai.networks.layers import Norm
from monai.networks.nets import FlexibleUNet, UNet
from torch.optim.lr_scheduler import MultiStepLR


class Net(L.LightningModule):
    def __init__(self):
        super().__init__()
        self._model = FlexibleUNet(
            in_channels=1,
            out_channels=1,
            backbone="resnet50",
            pretrained=True,
            spatial_dims=3,
        )

        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss()
        self.iou = MeanIoU(include_background=False)

    def forward(self, x):
        return torch.nn.functional.sigmoid(self._model(x))

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self._model.parameters(), 1e-4)
        lr_scheduler = MultiStepLR(optimizer, gamma=0.2, milestones=[200, 500])
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

    def training_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"]
        output = self.forward(images)
        l_dice = self.dice_loss(output, labels)
        l_focal = self.focal_loss(output, labels)
        total_loss = l_dice + l_focal
        self.log("train_loss", total_loss)
        self.log("train_dice_loss", l_dice)
        self.log("train_focal_loss", l_focal)
        return total_loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"]
        pred = self.forward(images)
        self.iou(y_pred=pred, y=labels)
        dice = self.dice_loss(pred, labels).detach().item()

        log_params = {"on_step": False, "on_epoch": True, "prog_bar": True}
        self.log("val_dice_loss", dice, **log_params)
        return {"val_dice_loss": dice}

    def on_validation_epoch_end(self):
        miou = self.iou.aggregate()
        self.iou.reset()
        self.log("val_miou", miou, prog_bar=True)


if __name__ == "__main__":
    model = Net()

    input = torch.rand(4, 1, 64, 64, 64)
    label = torch.rand(4, 1, 64, 64, 64)
    o = model.training_step({"image": input, "label": label}, 0)
    print(o)

    input = torch.rand(4, 1, 128, 128, 128)
    label = torch.rand(4, 1, 128, 128, 128)
    o = model.validation_step({"image": input, "label": label}, 0)
    print(o)
