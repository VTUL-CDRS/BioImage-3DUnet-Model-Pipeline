import os
import glob
from skimage import io
import numpy as np
import argparse

def combine(img_folder, term, outputfile):
    if os.path.exists(outputfile):
        raise Exception("output file already exists")
    imgs = glob.glob(os.path.join(img_folder, f"*{term}*"))
    imgs = sorted(imgs, key=os.path.getmtime)
    allimgs = io.imread(imgs[0])
    allimgs = np.reshape(allimgs, (1, *allimgs.shape))
    for i in range(1, len(imgs)):
        img = io.imread(imgs[i])
        img = np.reshape(img, (1, *img.shape))
        allimgs = np.concatenate((allimgs, img))
    io.imsave(outputfile, allimgs)

def main():
    parser = argparse.ArgumentParser(
        description="Combine images with the specified term in their filenames",
        usage="%(prog)s img_folder term outputfile\n\nExample: %(prog)s images/ *.tif testall.tif"
    )
    parser.add_argument("img_folder", help="The folder containing the images")
    parser.add_argument("term", help="The term to search for in image filenames")
    parser.add_argument("outputfile", help="The output file to save the combined images")

    args = parser.parse_args()

    combine(args.img_folder, args.term, args.outputfile)

if __name__ == "__main__":
    main()
