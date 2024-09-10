import zarr
import tifffile as tif 
from pathlib import Path
import tyro
import numpy as np

def main(input: str, output: str):
    input_dir = Path(input)
    output_dir = Path(output)

    gts_dir = input_dir / 'groundtruth'
    raw_dir = input_dir / 'raw_crops'

    for file in gts_dir.glob('crop*'):
        print(file)
        name = str(file).split('/')[-1]
        print(name)
        if name in ['crop48', 'crop109', 'crop32', 'crop40', 'crop110', 'crop39']:
            print('Skip')
            continue
        raw = zarr.open(raw_dir / f'{name}.zarr', mode='r')
        raw = np.array(raw)
        print(raw.dtype, raw.shape)

        gt = zarr.open(file / 'labels' / 'all', mode='r')
        gt = np.array(gt)[::2, ::2, ::2]
        print(gt.dtype, gt.shape)
        
        raw_file = output_dir / f'macrophage2_{name}_raw.tif'
        tif.imwrite(raw_file, raw)

        gt_file = output_dir / f'macrophage2_{name}_gt.tif'
        tif.imwrite(gt_file, gt, compression='zlib')

if __name__ == '__main__':
    tyro.cli(main)
