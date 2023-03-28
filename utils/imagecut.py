import argparse
import numpy as np
import tifffile


def load_3d_image(file_path):
    with tifffile.TiffFile(file_path) as tif:
        return tif.asarray()


def extract_sub_image(image, x, y, z, width, height, depth):
    return image[z:z+depth, y:y+height, x:x+width]


def main(input_path, output_path, x, y, z, width, height, depth):
    image = load_3d_image(input_path)
    sub_image = extract_sub_image(image, x, y, z, width, height, depth)
    tifffile.imwrite(output_path, sub_image)
    print("Cut image saved to", output_path)


if __name__ == "__main__":
    example_command = "python imagecut.py input_image.tiff output_image.tiff 100 200 50 300 400 100"
    parser = argparse.ArgumentParser(description="Extract a sub-image from a 3D tiff image\n\nExample command:\n" +
                                     example_command, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input_path", help="path to the input tiff image")
    parser.add_argument(
        "output_path", help="path to save the output tiff image")
    parser.add_argument(
        "x", type=int, help="x-coordinate of the top-left corner of the sub-image")
    parser.add_argument(
        "y", type=int, help="y-coordinate of the top-left corner of the sub-image")
    parser.add_argument(
        "z", type=int, help="z-coordinate of the top-left corner of the sub-image")
    parser.add_argument("width", type=int,
                        help="width of the sub-image in pixels")
    parser.add_argument("height", type=int,
                        help="height of the sub-image in pixels")
    parser.add_argument("depth", type=int,
                        help="depth of the sub-image in pixels")

    args = parser.parse_args()

    main(args.input_path, args.output_path, args.x, args.y,
         args.z, args.width, args.height, args.depth)

