from pathlib import Path

import numpy as np
import tifffile as tif
import tyro
from tqdm import tqdm


def main(input: str, output: str, step: int = 8):
    """
    input: input directory containing tif files
    output: filename for merged and downsampled tif file
    step: controls downsample rates
    """
    input_dir = Path(input)

    files = [str(f) for f in input_dir.glob("*.tif")]
    files = sorted(files)

    img = tif.imread(files[0])
    files = files[::step]
    H, W = img[::8, ::8].shape
    merged_img = np.zeros(shape=(len(files), H, W), dtype=np.uint8)
    for id, file in enumerate(tqdm(files)):
        img = tif.imread(file)
        img = img[::step, ::step]
        merged_img[id, ...] = img

    tif.imwrite(output, merged_img)


if __name__ == "__main__":
    tyro.cli(main)
