import os
import cv2
import glob
import numpy as np
from tqdm import tqdm
from scipy import stats
from skimage import data, io, filters, feature, transform, util, color
from scipy import ndimage as ndi

def load_and_preprocess_images(fonts, dataset_path):
    x_data = []
    for font in fonts:
        filenames = sorted(glob.glob(f'{dataset_path}/{font}/*.jpeg'))
        for filename in tqdm(filenames, desc=f"Processing {font}"):
            x_data.append(preprocess_image(filename))
    return np.asarray(x_data)

def preprocess_image(filename):
    img = io.imread(filename)
    filtered_img = filters.median(img, mode='nearest')
    gray_img = color.rgb2gray(filtered_img)
    binary_img = apply_threshold(gray_img)
    binary_img = util.img_as_ubyte(binary_img)
    inverted_binary_img = invert_image(binary_img)
    rotated_img = rotate_image(inverted_binary_img)
    return rotated_img

def apply_threshold(gray_img):
    thresh = filters.threshold_otsu(gray_img)
    return gray_img < thresh

def invert_image(binary_img):
    black_pixels = np.sum(binary_img == 0)
    total_pixels = binary_img.size
    percentage_black = (black_pixels / total_pixels) * 100
    if percentage_black > 50:
        return np.where(binary_img == 0, 255, 0)
    else:
        return binary_img

def rotate_image(inverted_binary_img):
    edges = feature.canny(inverted_binary_img, sigma=4)
    h, theta, d = transform.hough_line(edges)
    accum, angles, dists = transform.hough_line_peaks(h, theta, d)
    if len(angles) > 0:
        angles_deg = np.rad2deg(angles) + 90
        rotation_angle = stats.mode(angles_deg).mode
        rotated_img = transform.rotate(inverted_binary_img, rotation_angle, cval=1, resize=False)
        return util.img_as_ubyte(rotated_img)
    else:
        return inverted_binary_img

def generate_labels(fonts, dataset_path):
    font_to_num = {font: i for i, font in enumerate(fonts)}
    y_data = []
    for font in fonts:
        filenames = sorted(glob.glob(f'{dataset_path}/{font}/*.jpeg'))
        for filename in tqdm(filenames, desc=f"Processing {font}"):
            y_data.append(font_to_num[font])
    return np.asarray(y_data)

def save_images(images, labels, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for i, img in enumerate(images):
        img_uint8 = img.astype(np.uint8)
        filename = os.path.join(save_dir, f"{labels[i]}_{i}.jpg")
        cv2.imwrite(filename, img_uint8)

