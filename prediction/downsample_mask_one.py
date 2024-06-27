import tifffile as tif
import tyro


def main(file: str, mask: str):
    img = tif.imread(file)
    mask = tif.imread(mask)
    
    img = img[::2, ::2, ::2]
    mask = mask[::2, ::2, ::2]

    img = img * mask
    print(img.dtype, img.shape)

    outfile = file.split(".")[0] + "_downsampled_masked.tif"
    tif.imwrite(outfile, img, compression="zlib")


if __name__ == "__main__":
    tyro.cli(main)
