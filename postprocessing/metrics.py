import numpy as np
import tifffile as tif
import tyro


def dice(gt, pred):
    # Ensure binary masks
    gt = (gt > 0).astype(np.int32)
    pred = (pred > 0).astype(np.int32)

    # Flatten arrays
    gt = gt.flatten()
    pred = pred.flatten()

    intersection = np.sum(gt * pred)
    union = np.sum(gt) + np.sum(pred)

    # Compute Dice score
    return (2.0 * intersection + 1e-6) / (union + 1e-6)


def iou(gt, pred):
    # Ensure binary masks
    gt = (gt > 0).astype(np.int32)
    pred = (pred > 0).astype(np.int32)

    # Flatten arrays
    gt = gt.flatten()
    pred = pred.flatten()

    intersection = np.sum(gt * pred)
    union = np.sum(gt) + np.sum(pred) - intersection

    # Compute IoU
    return (intersection + 1e-6) / (union + 1e-6)


def main(gt_file: str, pred_file: str):
    gt = tif.imread(gt_file)
    pred = tif.imread(pred_file)

    pred = (pred > 127).astype(np.int32)
    print("Dice:", dice(gt, pred))
    print("IOU:", iou(gt, pred))


if __name__ == "__main__":
    tyro.cli(main)
