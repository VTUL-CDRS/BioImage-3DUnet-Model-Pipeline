import tifffile as tif
import tyro


def main(file: str):
    img = tif.imread(file)
    img = img[::2, ::2, ::2]

    print(img.dtype, img.shape)

    outfile = file.split(".")[0] + "_downsampled.tif"
    tif.imwrite(outfile, img, compression="zlib")


if __name__ == "__main__":
    tyro.cli(main)
