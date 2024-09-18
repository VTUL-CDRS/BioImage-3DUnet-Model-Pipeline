import numpy as np
import tifffile as tif
import torch
import tyro
from einops import rearrange
from model.net import Net
from monai.inferers import SlidingWindowInfererAdapt


def predict(
    ckpt: str, inputfile: str, outputfile: str, roi_size: int = 64, batch_size: int = 4
):
    cuda = torch.device("cuda:0")
    cpu = torch.device("cpu")
    model = Net.load_from_checkpoint(ckpt).to(cuda)
    model.eval()

    inputimage = tif.imread(inputfile)

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

        tif.imwrite(outputfile, output, compression="zlib")


if __name__ == "__main__":
    print("Predicting...")
    tyro.cli(predict)
    print("Done.")
