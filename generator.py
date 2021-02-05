#!/usr/bin/env python
import math
from keras.utils import Sequence
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
import cv2
import random
import imaug


def clamp(val, minimum, maximum):
    'Clamp value between a minimum and maximum'
    if val < minimum:
        return minimum
    if val > maximum:
        return maximum
    return val


def loadImage(path: str, target_size=None, grayscale=False):
    color_flag = cv2.IMREAD_COLOR if not grayscale else cv2.IMREAD_GRAYSCALE
    img = cv2.imread(path, color_flag)
    if target_size is not None:
        img = cv2.resize(img, target_size)
    return img


def getPadded(img: np.array, output_size: int):
    'Returns image padded so that height and width are the same'
    if img is None:
        return None
    if len(img.shape) == 3:
        h, w, _ = img.shape
        pixel_val = [0, 0, 0]
    elif len(img.shape) == 2:
        h, w = img.shape
        pixel_val = [0]
    else:
        raise ValueError('Unexpected image shape {}'.format(img.shape))
    top, bottom, left, right = 0, 0, 0, 0
    if h > w:
        left = math.floor((h-w)/2)
        right = math.ceil((h-w)/2)
    elif h < w:
        top = math.floor((w-h)/2)
        bottom = math.ceil((w-h)/2)
    padded = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=pixel_val) if h != w else img
    return cv2.resize(padded, (output_size, output_size))


def augment(img: np.array, vflip: bool, hflip: bool, rotation: int, translation_matrix: np.array):
    cols, rows = img.shape[:2]
    img = imaug.flip(img, vflip, hflip)
    img = imaug.rotate(img, rotation)
    if translation_matrix is not None:
        img = cv2.warpAffine(img, translation_matrix, (cols, rows))
    return img


class BatchGenerator(Sequence):
    'Batch generator class'

    def __init__(self, x_set, y_set, batch_size: int, shape: tuple, pad: bool = False, ops: list = [lambda a: a], grayscale: bool=False):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.shape = tuple(shape)
        self.pad = pad
        self.ops = ops
        self.grayscale = grayscale

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        #if self.pad:
        #    image_list = [getPadded(loadImage(f, grayscale=self.grayscale), max(self.shape[:2])) for f in batch_x if f is not None]
        #else:
        #    image_list = [loadImage(f, grayscale=self.grayscale, target_size=self.shape[:2]) for f in batch_x if f is not None]
        image_list = [cv2.resize(cv2.imread(f, cv2.IMREAD_COLOR), (224, 224)) for f in batch_x if f is not None]
        image_tensor = np.array(image_list).astype(np.float32)

        for op in self.ops:
            image_tensor = op(image_tensor)
        return image_tensor.astype(np.float32), np.array(batch_y)


class AugmentedBatchGenerator(Sequence):
    'Batch generator class with data augmentation'

    def __init__(self, x_set, y_set, batch_size: int, shape: tuple, pad: bool = False, ops: list = [lambda a: a], vflip: bool = True, hflip: bool = True, rotation_range: int = 360, grayscale=False, flip_probability: float = 0.5, x_translation_range: int = 1, y_translation_range: int = 1):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.shape = tuple(shape)
        self.pad = pad
        self.ops = ops
        self.grayscale = grayscale
        self.flip_prob = clamp(flip_probability, 0.0, 1.1)
        self.rotation_range = rotation_range
        self.vflip = vflip
        self.hflip = hflip
        self.x_translation_range = x_translation_range
        self.y_translation_range = y_translation_range
        self.i = 0

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        if self.pad:
            image_list = [getPadded(loadImage(f, grayscale=self.grayscale), max(self.shape[:2])) for f in batch_x if f is not None]
        else:
            image_list = [loadImage(f, grayscale=self.grayscale, target_size=self.shape[:2]) for f in batch_x if f is not None]
        auglist = []
        for img in image_list:
            # augment
            vflip = (random.uniform(0, 1) > self.flip_prob) if self.vflip else False
            hflip = (random.uniform(0, 1) > self.flip_prob) if self.hflip else False
            rotation_angle = random.randrange(0, self.rotation_range)

            y_trans = random.randint(1, self.y_translation_range)
            x_trans = random.randint(1, self.x_translation_range)

            M = np.float32([[1, 0, y_trans], [0, 1, x_trans]])
            a = augment(img, vflip, hflip, rotation_angle, M)
            auglist.append(a)

        image_tensor = np.array(auglist).astype(np.float32)
        for op in self.ops:
            image_tensor = op(image_tensor)
        return image_tensor.astype(np.float32), np.array(batch_y)
