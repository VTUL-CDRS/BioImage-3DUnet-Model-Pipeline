from pathlib import Path

import numpy as np
import tifffile as tif
import torch
import tyro
from einops import rearrange
from model.net import Net
from monai.inferers import SlidingWindowInfererAdapt


def predict_list(
    ckpt: str, inputfile: str, outputfile: str, roi_size: int = 64, batch_size: int = 4
):
    cuda = torch.device("cuda:0")
    cpu = torch.device("cpu")
    model = Net.load_from_checkpoint(ckpt).to(cuda)
    model.eval()

    input_dir = Path(inputfile)
    output_dir = Path(outputfile)
    files = [str(f) for f in input_dir.glob("*.tif")]
    files = sorted(files)
    for i in range(0, len(files), roi_size):
        print(f"{i}/{len(files)}")
        if i + roi_size >= len(files):
            continue

        imgs = []
        for j in range(roi_size):
            img = tif.imread(files[i + j])
            imgs.append(img)
        inputimage = np.stack(imgs)

        input = torch.tensor(inputimage, dtype=torch.float, device=cpu)
        input = input / 255.0

        inferer = SlidingWindowInfererAdapt(
            roi_size=(roi_size, roi_size, roi_size),
            sw_batch_size=batch_size,
            sw_device=cuda,
            progress=True,
        )

        input = rearrange(input, "z x y -> 1 1 z x y")

        with torch.no_grad():
            output = inferer(input, model)

            output = rearrange(output, "1 1 z x y -> z x y")
            output = output.cpu().numpy()
            output = (255 * output).astype(np.uint8)

            name = files[i].split("/")[-1]
            tif.imwrite(output_dir / name, output, compression="zlib")


if __name__ == "__main__":
    print("Predicting...")
    tyro.cli(predict_list)
    print("Done.")
