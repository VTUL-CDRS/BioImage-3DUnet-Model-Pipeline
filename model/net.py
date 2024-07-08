import lightning as L
import torch
from monai.inferers import sliding_window_inference
from monai.losses import DiceLoss, FocalLoss
from monai.metrics import MeanIoU
from monai.networks.layers import Norm
from monai.networks.nets import FlexibleUNet, UNet


class Net(L.LightningModule):
    def __init__(self):
        super().__init__()
        # self._model = UNet(
        #     spatial_dims=3,
        #     in_channels=1,
        #     out_channels=1,
        #     channels=(16, 32, 64, 128, 256),
        #     strides=(2, 2, 2, 2),
        #     num_res_units=2,
        #     norm=Norm.BATCH,
        # )

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
        return self._model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self._model.parameters(), 1e-4)
        return optimizer

    def training_step(self, batch, batch_idx):
        print(len(batch), batch_idx)
        images, labels = batch[batch_idx]["image"], batch[batch_idx]["label"]
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
        roi_size = (64, 64, 64)
        sw_batch_size = 4
        outputs = sliding_window_inference(
            images, roi_size, sw_batch_size, self.forward
        )
        self.iou(y_pred=outputs, y=labels)
        iou = self.iou.aggregate().item()
        self.iou.reset()
        dice = self.dice_loss(outputs, labels)
        log_params = {"on_step": False, "on_epoch": True, "prog_bar": True}
        self.log("val_dice_loss", dice, **log_params)
        self.log("val_iou", iou, **log_params)
        return {"val_dice_loss": dice, "val_iou": iou}


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
