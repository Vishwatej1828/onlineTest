import cv2 as cv
import numpy as np
import os
from glob import glob

input_dir = 'input_data'
output_dir = 'input_data\__MACOSX'

os.makedirs(output_dir, exist_ok=True)

def applyMask(file_path):
    image = cv.imread(file_path)

    if image is None:
        print(f"Failed to read the image: {file_path}")
        return 0

    # Create a binary mask where all 3 channels pixel values are above 200
    mask = cv.inRange(image, (200, 200, 200), (255, 255, 255))

    # number of max-value pixels in the image
    No_of_Max_Pixels_in_Image = np.sum(mask == 255)

    # Save the mask
    output_path = os.path.join(output_dir, "._" + os.path.basename(file_path))
    cv.imwrite(output_path, mask)

    # return the max_pixel count of the particular image
    return No_of_Max_Pixels_in_Image

def main():
    # Get list of all .jpg & .png files in the input directory
    jpg_files = glob(os.path.join(input_dir, "*.jpg"))
    png_files = glob(os.path.join(input_dir, "*.png"))

    total_files = jpg_files + png_files
    total_max_pixel_count = 0

    for file_path in total_files:
        total_max_pixel_count += applyMask(file_path)

    # print total number of max pixels in all images
    print(f"Total number of max pixels in all images: {total_max_pixel_count}")

if __name__ == '__main__':
    main()
