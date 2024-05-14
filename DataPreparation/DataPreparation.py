import os
import glob
import skimage as ski
import cv2
import numpy as np
from tqdm import tqdm

def preprocess_image(img):
    filtered_img      = ski.filters.median(img, mode = 'nearest')
    gray_scaled_img   = ski.color.rgb2gray(filtered_img)

    # Calculate the Otsu's threshold for the grayscale image
    threshold = ski.filters.threshold_otsu(gray_scaled_img)

    # Create a binary image by setting all pixels with a value less than the threshold to True (foreground) 
    # and others to False (background)
    binary_img = gray_scaled_img < threshold

    # Convert the binary image to an 8-bit unsigned integer format for further processing
    # In this format, each pixel value ranges from 0 to 255, where 0 represents black and 255 represents white.
    ubyte_img = img = ski.util.img_as_ubyte(binary_img)

    # Calculate the number of black pixels in the ubyte image (pixels with a value of 0)
    black_pixels = np.sum(ubyte_img == 0)

    # Get the total number of pixels in the ubyte image
    total_pixels = ubyte_img.size

    # Calculate the percentage of black pixels in the image
    percentage_black_pixels = (black_pixels / total_pixels) * 100

    # If more than 50% of the pixels in the image are black
    if percentage_black_pixels > 50:
        # Invert the image colors (i.e., black becomes white and vice versa)
        white_background_img = np.where(ubyte_img == 0, 255, 0)
    else:
        # If less than or equal to 50% of the pixels are black, keep the image as is
        white_background_img = ubyte_img  

    return white_background_img



# Mimicing the idea of pagination, the following function will preprocess and save the images in batches, given a start index and a batch size
def preprocess_and_save_images(start_index = 0, batch_size = 1000):
    Fonts  = [ 'IBM Plex Sans Arabic', 'Lemonada', 'Marhey','Scheherazade New'] 

    # Define the directory where the preprocessed images will be saved
    base_save_dir = "C:\\Users\\fatom\\Documents\\Pattern Labs\\Group-Project\\arabic-font-recognition\\Data\\preprocessed_images_3"

    # Check if the base directory exists. If it doesn't, create a new directory with that name
    if not os.path.exists(base_save_dir):
        os.makedirs(base_save_dir)

    for font in Fonts:
        font_path = f"C:\\Users\\fatom\\Documents\\Pattern Labs\\Group-Project\\arabic-font-recognition\\Data\\fonts-dataset\\{font}\\*.jpeg"

        # Create a directory for the current font inside the base directory
        font_save_dir = os.path.join(base_save_dir, font)
        if not os.path.exists(font_save_dir):
            os.makedirs(font_save_dir)

        # Get all files in the font_path directory, sorted in ascending order
        all_files = sorted(glob.glob(font_path))

        # Determine the end index for processing
        end_index = min(start_index + batch_size, len(all_files))

        # Process only the files within the start and end indices
        for i, filename in enumerate(tqdm(all_files[start_index:end_index], desc=f"Processing {font}")):
            img = ski.io.imread(filename)
            white_background_img = preprocess_image(img)          

            # Save the processed image immediately
            filename = os.path.join(font_save_dir, f"{str(i + start_index).zfill(3)}.jpg")
            cv2.imwrite(filename, white_background_img)

    print("Image preprocessing and saving completed.")



def load_images(start_index = 0, batch_size = 1000):
    # Define the directory where the preprocessed images are stored
    base_img_dir = "C:\\Users\\fatom\\Documents\\Pattern Labs\\Group-Project\\arabic-font-recognition\\Data\\preprocessed_images_3"

    # Retrieve the list of all font directories in the base directory
    font_dirs = os.listdir(base_img_dir)

    # Initialize two empty lists to store the image data and corresponding labels
    x_data = []
    y_data = []

    # Iterate over each font directory
    for font_dir in tqdm(font_dirs, desc="Loading fonts"):
        # Construct the full path for the current font directory
        font_path = os.path.join(base_img_dir, font_dir)
        # Retrieve the list of all image files in the current font directory
        img_files = sorted(os.listdir(font_path))
        # Calculate the end index for the current batch
        end_index = start_index + batch_size

        # Iterate over each image file in the current batch
        for i in range(start_index, min(end_index, len(img_files))):
            # Construct the full file path for the current image file
            img_path = os.path.join(font_path, img_files[i])

            # Read the image file as a grayscale image using OpenCV
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            # Add the image data to the x_data list
            x_data.append(img)

            # Add the font name as the label to the y_data list
            y_data.append(font_dir)

    # Convert the lists of image data and labels to numpy arrays for easier manipulation
    x_data = np.array(x_data)
    y_data = np.array(y_data)

    # Randomly shuffle the data and labels to ensure a good mix for training
    p = np.random.permutation(len(x_data))
    x_data = x_data[p]
    y_data = y_data[p]

    return x_data, y_data