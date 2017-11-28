# -*- coding: utf-8 -*-


import cv2
import numpy as np
from os import listdir, sep
from math import pi
from os import path
import matplotlib.pyplot as plt

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
fat_max_area = 2000
#   circularity
fat_flt_by_circularity = True
fat_min_circularity = 0.6
#   color of detection
fat_detector_color = (125, 0, 0)
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


def measure_fat_area(image, visualize_detection=False):
    """

    :param image:
    :param visualize_detection:
    :return: area, if visualize_detection True - (area, image_with_detection)
    """
    detector = get_fat_detector()
    detection = search_fat(detector, image, visualize_detection)
    return detection


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


def search_fat(detector, image, visualize_detection=False):
    keypoints = detector.detect(image)
    area = 0
    for k in keypoints:
        area += (pow(k.size, 2) / 4) * pi
    if visualize_detection:
        image_with_detections = cv2.drawKeypoints(image, keypoints, np.array([]),
                                                  fat_detector_color, cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        return area, image_with_detections
    return area


def calculate_fat_percent(image, fat_area, background_area):
    normal_tissue = cv2.countNonZero(image) - background_area
    return 100 * fat_area / normal_tissue


def predict_and_visualize(path_to_image):
    image, background_area = preprocess_image_and_measure_background_area(path_to_image)
    fat_area, detection_image = measure_fat_area(image, visualize_detection=True)
    fat_percent = calculate_fat_percent(image, fat_area, background_area)
    if fat_percent >= fat_threshold:
        prediction = fatty
    else:
        prediction = normal
    print(path_to_image, "\n", "Class: ", prediction, "\n", "Fat percent: ", fat_percent)
    return detection_image


def measure_fat_percentages_in_folder(folder):
    images = get_images_from_folder(folder)
    fat_percentages = [predict(image)[1] for image in images]
    return fat_percentages


def get_images_from_folder(folder):
    files = []
    filenames = listdir(folder)
    for file in filenames:
        for ext in images_extension:
            if file.endswith(ext):
                files.append(path.join(folder, file))
                break
    return files


def test_accuracy(fat_dataset, normal_dataset):
    correct = 0
    for example in fat_dataset:
        if predict(example)[0] == fatty:
            correct += 1
    for example in normal_dataset:
        if predict(example)[0] == normal:
            correct += 1
    return correct / (len(fat_dataset) + len(normal_dataset))

