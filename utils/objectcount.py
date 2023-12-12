import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse

def count_objects(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur
    blur = cv2.GaussianBlur(gray, (11, 11), 0)

    # Detect edges using Canny edge detection
    canny = cv2.Canny(blur, 30, 150, 3)

    # Dilate the edges
    dilated = cv2.dilate(canny, (1, 1), iterations=0)

    # Find contours
    (cnt, hierarchy) = cv2.findContours(
        dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    return len(cnt)

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Count objects in an image')
    parser.add_argument('image', help='Path to the image file')
    args = parser.parse_args()

    # Call the count_coins function and print the result
    num_coins = count_objects(args.image)
    print("Objects in the image:", num_coins)
