#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import cv2
from shapely.geometry import Polygon
from PIL import Image
import random
from os import listdir
from os import path
from os import makedirs
from random import choice
import argparse
import json

max_w, max_h = 512, 512
test_count, train_count = 1000, 10000


def in_rectangle(width, height, x, y):
    if 0 < x < width and height > y > 0:
        return True
    return False


def poly_in_rectangle(points, width, height):
    for p in points:
        if not in_rectangle(width, height, p[0], p[1]):
            return False
    return True


def valid_intersection_polygons(i0, i1, i2, i3, max_x, max_y):
    inner_polygon = Polygon([(i0[0], i0[1]), (i1[0], i1[1]), (i2[0], i2[1]),
                             (i3[0], i3[1])])
    outer_polygon = max_x * max_y
    p = inner_polygon.area / outer_polygon * 100
    if p > 0.5:
        return True
    return False


def point_rotare(point, matrix):
    r = matrix.dot(np.float32([[point[0]], [point[1]], [1]]))
    return r[0][0], r[1][0]


def find_centre(points):
    left_x, top_y, right_x, bottom_y = bounding_box(points)
    return (left_x + right_x) / 2, (top_y + bottom_y) / 2


def rotate_and_scale_points(points, angle_min, angle_max, scale_min, scale_max,
                            img_width, img_height):
    angle = random.uniform(angle_min, angle_max)
    while True:
        scale = random.uniform(scale_min, scale_max)
        centre = find_centre(points)
        M = cv2.getRotationMatrix2D(centre, angle, scale)
        new_points = list(map(lambda x: point_rotare(x, M), points))
        return new_points


def bounding_box(points):
    left_x = min(points, key=lambda p: p[0])[0]
    right_x = max(points, key=lambda p: p[0])[0]
    bottom_y = min(points, key=lambda p: p[1])[1]
    top_y = max(points, key=lambda p: p[1])[1]
    return (left_x, top_y, right_x, bottom_y)


def move_poly(points, img_width, img_height):
    left_x, top_y, right_x, bottom_y = bounding_box(points)
    dx = random.uniform(-left_x, img_width - right_x)
    dy = random.uniform(-bottom_y, img_height - top_y)
    new_points = list(map(lambda p: [p[0] + dx, p[1] + dy], points))
    return new_points


def random_shift_point(point, d, img_width, img_height):
    x = random.uniform(max(0, point[0] - d), min(img_width, point[0] + d))
    y = random.uniform(max(0, point[1] - d), min(img_height, point[1] + d))
    return [x, y]


def random_perspective_points_transform(points, value, img_width, img_height):
    left_x, top_y, right_x, bottom_y = bounding_box(points)
    d = (right_x - left_x + top_y - bottom_y) / 2 * value
    new_points = list(
        map(lambda p: random_shift_point(p, d, img_width, img_height), points))
    return new_points


def gen_points():
    x0, y0 = 0, 0
    x1, y1 = x0 + max_w, y0
    x2, y2 = x0 + max_w, y0 + max_h
    x3, y3 = x0, y0 + max_h
    return x0, y0, x1, y1, x2, y2, x3, y3


def genM(img, augmentation_settings):
    angle_min = augmentation_settings['angle_min']
    angle_max = augmentation_settings['angle_max']
    zoom_min = augmentation_settings['zoom_min']
    zoom_max = augmentation_settings['zoom_max']
    persp_k = augmentation_settings['persp_k']
    x0, y0, x1, y1, x2, y2, x3, y3 = gen_points()
    rows, cols, ch = img.shape
    orig_points = np.float32([[x0, y0], [x1, y1], [x2, y2], [x3, y3]])
    new_points = orig_points
    new_points = rotate_and_scale_points(new_points, angle_min, angle_max,
                                         zoom_min, zoom_max, cols, rows)
    new_points = move_poly(new_points, cols, rows)
    new_points = random_perspective_points_transform(new_points, persp_k, cols,
                                                     rows)
    new_points = np.float32(new_points)
    M = cv2.getPerspectiveTransform(new_points, orig_points)
    return M, new_points


def save_tile(img, M, save_img_path, color=True):
    if color:
        transformed_img = cv2.cvtColor(
            cv2.warpPerspective(img, M, (max_w, max_h)), cv2.COLOR_BGR2RGB)
    else:
        transformed_img = cv2.cvtColor(
            cv2.warpPerspective(img, M, (max_w, max_h)), cv2.COLOR_GRAY2RGB)
    im_pil = Image.fromarray(transformed_img)
    im_pil.save(save_img_path)
    return transformed_img


def count_colours(img):
    unique, counts = np.unique(img, return_counts=True)
    return dict(zip(unique, counts))


def threshold_black(img, threshold):
    result = count_colours(img)
    try:
        if result[255] < (result[0] + result[255]) * threshold:
            return True
        return False
    except:
        return False


def crop(img_path,
         img_bg_path,
         img_mask_path,
         save_img_path,
         save_bg_path,
         augmentation_settings,
         threshold=0.5):
    img = cv2.imread(img_path)
    M, new_points = genM(img, augmentation_settings)
    if path.exists(img_mask_path):
        img_mask = cv2.imread(img_mask_path, cv2.IMREAD_UNCHANGED)[:, :, 3]
        transformed_img = cv2.cvtColor(
            cv2.warpPerspective(img_mask, M, (max_w, max_h)),
            cv2.COLOR_GRAY2RGB)
        # Тут проверка на точки через пороговое значение
        if threshold_black(transformed_img, threshold):
            return False
    img_bg = cv2.imread(img_bg_path, cv2.IMREAD_UNCHANGED)[:, :, 3]

    colored_img = random_color(img, augmentation_settings)
    _ = save_tile(colored_img, M, save_img_path, True)
    tile_img_bg = save_tile(img_bg, M, save_bg_path, False)

    write_log(save_bg_path[:-4] + '.txt', tile_img_bg, new_points)

    return True


def get_jpg_files_whithout_background(folder_path):
    files = []
    for file in listdir(folder_path):
        if file.endswith('.png') and '_bg' not in file and '_mask' not in file:
            files.append(path.join(folder_path, file))
    return files


def cropping(root_folder, save_folder, max_files, augmentation_settings):
    if not path.exists(save_folder):
        makedirs(save_folder)

    without_background = get_jpg_files_whithout_background(root_folder)
    count = 0
    while count < max_files:
        image = choice(without_background)
        base_name = path.basename(image)

        mask_fat_name = base_name[:-4] + '_mask.png'
        mask_fat_name_full = path.join(image[:-len(base_name)], mask_fat_name)

        bg_name = base_name[:-4] + '_bg.png'
        bg_name_full = path.join(image[:-len(base_name)], bg_name)

        save_img_path = path.join(save_folder, str(count) + '_' + base_name)
        save_bg_path = path.join(save_folder, str(count) + '_' + bg_name)

        print("Paths:", image, bg_name_full, mask_fat_name_full, save_img_path,
              save_bg_path)
        if crop(
                image,
                bg_name_full,
                mask_fat_name_full,
                save_img_path,
                save_bg_path,
                augmentation_settings,
                threshold=0.5):
            count += 1


def random_color(img, augmentation_settings):
    dy_max = augmentation_settings['dy_max']
    dcb_max = augmentation_settings['dcb_max']
    dcr_max = augmentation_settings['dcr_max']
    ycrcb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    dy = random.randint(-dy_max, dy_max)
    dcb = random.randint(-dcb_max, dcb_max)
    dcr = random.randint(-dcr_max, dcr_max)
    y = (np.clip(y.astype(np.int32) + dy, 0, 255)).astype(np.uint8)
    cr = (np.clip(cr.astype(np.int32) + dcr, 0, 255)).astype(np.uint8)
    cb = (np.clip(cb.astype(np.int32) + dcb, 0, 255)).astype(np.uint8)
    ycrcb = cv2.merge((y, cr, cb))
    img = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)
    return img


def write_log(name, img, points):
    colours = count_colours(img)
    answer = 0
    if 0 in colours:
        answer = 0
    if 255 in colours:
        answer = 1
    if 255 in colours and 0 in colours:
        answer = colours[255] / (colours[0] + colours[255])

    f = open(name, 'w')
    f.write(json.dumps(points.tolist()))
    f.write('\n')
    f.write(str(answer))
    f.close()


def smart_crop():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i', '--input', help='Input folder name', required=True)
    parser.add_argument(
        '-o', '--output', help='Output folder name', required=True)
    args = parser.parse_args()

    train_folder = path.join(args.output, 'train')
    if not path.exists(train_folder):
        makedirs(train_folder)

    test_folder = path.join(args.output, 'test')
    if not path.exists(test_folder):
        makedirs(test_folder)

    train_augmentation = {
        'dy_max': 10,
        'dcb_max': 20,
        'dcr_max': 20,
        'angle_min': 0,
        'angle_max': 360,
        'zoom_min': 0.8,
        'zoom_max': 1.2,
        'persp_k': 0.1
    }

    test_augmentation = {
        'dy_max': 0,
        'dcb_max': 0,
        'dcr_max': 0,
        'angle_min': 0,
        'angle_max': 360,
        'zoom_min': 1.0,
        'zoom_max': 1.0,
        'persp_k': 0.0
    }

    cropping(
        path.join(args.input, 'test_bad'), path.join(test_folder, 'bad'),
        test_count, test_augmentation)
    cropping(
        path.join(args.input, 'test_good'), path.join(test_folder, 'good'),
        test_count, test_augmentation)
    cropping(
        path.join(args.input, 'train_bad'), path.join(train_folder, 'bad'),
        train_count, train_augmentation)
    cropping(
        path.join(args.input, 'train_good'), path.join(train_folder, 'good'),
        train_count, train_augmentation)


if __name__ == '__main__':
    smart_crop()
