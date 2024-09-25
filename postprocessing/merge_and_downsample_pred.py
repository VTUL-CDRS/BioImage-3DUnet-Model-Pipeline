from pathlib import Path

import numpy as np
import tifffile as tif
import tyro
from tqdm import tqdm
from einops import rearrange

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
    D, H, W = img[::8, ::8, ::8].shape
    merged_img = np.zeros(shape=(len(files), D, H, W), dtype=np.uint8)
    for id, file in enumerate(tqdm(files)):
        img = tif.imread(file)
        img = img[::step, ::step, ::step]
        merged_img[id, ...] = img

    merged_img = rearrange(merged_img, 'n z x y -> (n z) x y')
    tif.imwrite(output, merged_img, compression='zlib')


if __name__ == "__main__":
    tyro.cli(main)
