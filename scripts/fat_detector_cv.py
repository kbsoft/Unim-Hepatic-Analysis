# -*- coding: utf-8 -*-


import cv2
import numpy as np
from os import listdir
import os
from math import pi
from os import path
import argparse
import subprocess
from shutil import copyfile

images_extension = ['.png', '.jpg', '.jpeg']
fatty = 'bad'
normal = 'good'
fat_threshold = 5

#   preprocess parameters for background detection
#   gaussian
background_kernel_size = (0, 0)
background_sigma = 60  # 60
#   binary converter
background_min_threshold = 220
background_max_value = 255

#   preprocess parameters for fat detection
#   gaussian
fat_kernel_size = (0, 0)
fat_sigma = 4
#   binary converter
fat_min_threshold = 180
fat_max_value = 255

#   fat blob detector parameters
#   filter by area
fat_flt_by_area = True
fat_min_area = 100
fat_max_area = 5000
#   circularity
fat_flt_by_circularity = True
fat_min_circularity = 0.6
#   color of detection
fat_detector_color = (0, 0, 0)
#   inertia
fat_flt_by_inertia = True
fat_min_inertia = 0.2  # 0 - line, 1 - circle
#   convexity
fat_flt_by_convex = False
fat_minConvexity = 0.95


def predict(path_to_image):
    """

    :param path_to_image:
    :return: (image classification, percent of fat on the image)
    """
    image, background_area = preprocess_image_and_measure_background_area(path_to_image)
    fat_area = measure_fat_area(image)
    fat_percent = calculate_fat_percent(image, fat_area, background_area)
    if fat_percent >= fat_threshold:
        prediction = fatty
    else:
        prediction = normal
    return prediction, fat_percent


def preprocess_image_and_measure_background_area(path_to_image):
    """

    :param path_to_image:
    :return: (preprocessed image, area of background)
    """
    image = cv2.imread(path_to_image)

    #   grayscale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #   mask background
    image, background_area = fill_in_background_and_get_its_area(image)

    #   histogram equalization
    image = cv2.equalizeHist(image)

    #   blurring
    image = cv2.GaussianBlur(image, fat_kernel_size, fat_sigma)

    #   convert to binary
    image = cv2.threshold(image, fat_min_threshold, fat_max_value, cv2.THRESH_BINARY)[1]

    #   negative
    image = cv2.bitwise_not(image)

    return image, background_area


def fill_in_background_and_get_its_area(image):
    background = get_background(image)

    number_background_pixels = cv2.countNonZero(background)

    average_pixel_value = image.mean()
    image_area = get_image_area(image)
    average_pixel_value_without_background = (average_pixel_value * image_area - number_background_pixels * 255) / \
                                             (image_area - number_background_pixels)

    inverted_background = cv2.bitwise_not(background)
    mask = cv2.add(inverted_background, average_pixel_value_without_background)
    masked_background_image = image + mask
    return masked_background_image, number_background_pixels


def get_background(image):
    blurred_image = cv2.GaussianBlur(image, background_kernel_size, background_sigma)
    binary_background = cv2.threshold(blurred_image, background_min_threshold, background_max_value,
                                      cv2.THRESH_BINARY)[1]
    return binary_background


def get_image_area(image):
    h, w = image.shape[:2]
    return h * w


def measure_fat_area(image, get_fat_keypoints=False):
    """

    :param image:
    :param get_fat_keypoints:
    :return: area, if get_fat_keypoints True - (area, fat_keypoints)
    """
    detector = get_fat_detector()
    fat_keypoints = search_fat(detector, image)
    fat_area = 0
    for k in fat_keypoints:
        fat_area += (pow(k.size, 2) / 4) * pi
    if get_fat_keypoints:
        return fat_area, fat_keypoints
    return fat_area


def get_fat_detector():
    params = cv2.SimpleBlobDetector_Params()

    #   area
    params.filterByArea = fat_flt_by_area
    params.minArea = fat_min_area
    params.maxArea = fat_max_area

    #   circle
    params.filterByCircularity = fat_flt_by_circularity
    params.minCircularity = fat_min_circularity

    #   inertia
    params.filterByInertia = fat_flt_by_inertia
    params.minInertiaRatio = fat_min_inertia

    #   convex
    params.filterByConvexity = fat_flt_by_convex
    params.minConvexity = fat_minConvexity

    detector = cv2.SimpleBlobDetector_create(params)
    return detector


def search_fat(detector, image):
    keypoints = detector.detect(image)
    return keypoints


def calculate_fat_percent(image, fat_area, background_area):
    normal_tissue = cv2.countNonZero(image) - background_area
    return 100 * fat_area / normal_tissue


def predict_and_get_full_info(path_to_image):
    binary_image, background_area = preprocess_image_and_measure_background_area(path_to_image)
    fat_area, fat_keypoints = measure_fat_area(binary_image, get_fat_keypoints=True)
    fat_percent = calculate_fat_percent(binary_image, fat_area, background_area)
    if fat_percent >= fat_threshold:
        prediction = fatty
    else:
        prediction = normal

    source_image = cv2.imread(path_to_image)
    fade_binary_image = np.uint8(binary_image * 0.2 + 0.8*255)
    binary_image_with_detection = cv2.drawKeypoints(fade_binary_image, fat_keypoints, np.array([]), fat_detector_color,
                                                   cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    fade_source_image = np.uint8(source_image * 0.6 + 0.4*255)
    source_image_with_detection = cv2.drawKeypoints(fade_source_image, fat_keypoints, np.array([]), fat_detector_color,
                                                   cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    return prediction, fat_percent, source_image, binary_image_with_detection, source_image_with_detection


def predict_and_visualize(path_to_image):
    binary_image, background_area = preprocess_image_and_measure_background_area(path_to_image)
    fat_area, fat_keypoints = measure_fat_area(binary_image, get_fat_keypoints=True)
    fat_percent = calculate_fat_percent(binary_image, fat_area, background_area)
    if fat_percent >= fat_threshold:
        prediction = fatty
    else:
        prediction = normal
    fade_binary_image = np.uint8(binary_image * 0.2 + 0.8 * 255)
    detection_image = cv2.drawKeypoints(fade_binary_image, fat_keypoints, np.array([]), fat_detector_color,
                                                    cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    print(path_to_image, "\n", "Class: ", prediction, "\n", "Fat percent: ", fat_percent)
    return detection_image


def get_images_from_folder(folder):
    files = []
    filenames = listdir(folder)
    for file in filenames:
        for ext in images_extension:
            if file.endswith(ext):
                files.append(path.join(folder, file))
                break
    return files


def accuracy_test(fat_dataset, normal_dataset):
    correct = 0
    for example in fat_dataset:
        if predict(example)[0] == fatty:
            correct += 1
    for example in normal_dataset:
        if predict(example)[0] == normal:
            correct += 1
    return correct / (len(fat_dataset) + len(normal_dataset))


def get_git_revision_hash():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip().decode()


def main():
    parser = argparse.ArgumentParser(description='Unim test task')
    subparsers = parser.add_subparsers(help='select what to do')
    subparsers.required = True

    # create the parser for the "predict" command
    predict_parser = subparsers.add_parser('predict', help='predict on directory')
    predict_parser.add_argument(
        '-i', '--input', help='Input folder name', required=True)
    predict_parser.add_argument(
        '-o', '--output', help='Output folder name', required=True)

    args = parser.parse_args()

    assert os.path.exists(args.input)
    files = get_images_from_folder(args.input)
    assert len(files)
    output_directory = args.output
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    good_fld = os.path.join(output_directory, 'good')
    bad_fld = os.path.join(output_directory, 'bad')
    if not os.path.exists(good_fld):
        os.makedirs(good_fld)
    if not os.path.exists(bad_fld):
        os.makedirs(bad_fld)
    for file in files:
        prediction, fat_percent, _, bin_img, _ = predict_and_get_full_info(file)
        name_with_extension = os.path.basename(file)
        name, ext = os.path.splitext(name_with_extension)

        if prediction == 'good':
            folder = good_fld
        else:
            folder = bad_fld
        copyfile(file, os.path.join(folder, name_with_extension))
        cv2.imwrite(os.path.join(folder, name + '_detect' + ext), bin_img)
        with open(os.path.join(folder, name + '.txt'), 'w') as info_file:
            info_file.write("Fat percent: {:.2f}".format(fat_percent))


if __name__ == "__main__":
    main()
