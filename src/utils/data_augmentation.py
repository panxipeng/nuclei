import numpy as np # linear algebra
import random
import skimage.io
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
import os
from dirs import ROOT_DIR
from src.utils import data_exploration as de

SPLIT_SIZE = 128
OVERLAP = 40
IMG_CHANNELS = 3


def read_train_image_labels(image_id):
    # most of the content in this function is taken from 'Example Metric Implementation' kernel
    # by 'William Cukierski'
    image_file = os.path.join(ROOT_DIR, r"data/images/train/{}/images/{}.png".format(image_id,image_id))
    mask_file = os.path.join(ROOT_DIR, r"data/images/train/{}/masks/*.png".format(image_id))
    image = skimage.io.imread(image_file)[:, :, :IMG_CHANNELS]
    masks = skimage.io.imread_collection(mask_file).concatenate()
    height, width, _ = image.shape
    labels = np.zeros((height, width, 1), dtype=np.bool)
    for id, mask_file in enumerate(masks):
        mask_ = np.expand_dims(mask_file, axis=-1)
        labels = np.maximum(labels, mask_)


    return image, labels


def read_test_image(image_id):
    image_file = os.path.join(ROOT_DIR, r"data/images/test/{}/images/{}.png".format(image_id, image_id))
    image = skimage.io.imread(image_file)[:, :, :IMG_CHANNELS]

    return image


"""
                    PHYSICAL AUGMENTATIONS
                   (SPLIT, ROTATE, RESCALE)
"""


def calculate_split_parameters(width, height, split_size=128, overlap=40):
    """
    Function calculates minimal shape of image, which divides by split size without remainder, and
    amount of small images (with size split_size x split_size) which we could get from the source image
    :param width: Width of the source image
    :param height: Height of the source image
    :param split_size: Size of the small image
    :param overlap: Overlapping pixels amount
    :return: new_shape, images_amount
    """
    new_shape = []
    images_amount = []
    for idx, x in enumerate([width, height]):
        n = 0
        while True:
            result = split_size + (split_size - overlap) * n
            if (result - x) > 0:
                break
            n += 1
        images_amount.append(n + 1)
        new_shape.append(result)

    return new_shape, images_amount



def split(image, labels=None, split_size=SPLIT_SIZE, overlap=OVERLAP):
    """
    Function to split images into bunch of smaller images with size (split_size x split_size)
    :param image: Input image
    :param labels: Input mask (if needed)
    :param split_size: Size of the small images (important: split_size % 16 == 0)
    :param overlap: Overlapping pixels amount
    :return: splited image and labels (if needed)
    """

    # Calculate new shape of image
    new_shape, images_amount = calculate_split_parameters(image.shape[0], image.shape[1], split_size, overlap)

    # Calculate pads list and reshape image
    width_diff = new_shape[0] - image.shape[0]
    if width_diff % 2 == 0:
        left_pad = right_pad = int((new_shape[0] - image.shape[0])/2)
    else:
        left_pad = int((new_shape[0] - image.shape[0])/2)
        right_pad = int((new_shape[0] - image.shape[0]) / 2) + 1

    height_diff = new_shape[1] - image.shape[1]
    if height_diff % 2 == 0:
        top_pad = bottom_pad = int((new_shape[1] - image.shape[1])/2)
    else:
        top_pad = int((new_shape[1] - image.shape[1]) / 2)
        bottom_pad = int((new_shape[1] - image.shape[1]) / 2) + 1

    pads = ((left_pad, right_pad), (top_pad, bottom_pad))

    # Split function
    split_step = split_size - overlap
    image_split_amount = images_amount[0] * images_amount[1]

    def split_func(img):
        image_split = np.ndarray((image_split_amount, split_size, split_size), dtype=np.uint8)
        count = 0
        for i in range(0, new_shape[0] - overlap, split_step):
            for j in range(0, new_shape[1] - overlap, split_step):
                image_split[count, :, :] = img[i:i + split_size, j:j + split_size]
                count += 1

        return image_split

    # Splitting image and labels

    if image.ndim == 3 and image.shape[2] != 1:
        reshaped_image = np.ndarray((new_shape[0], new_shape[1], image.shape[2]), dtype=np.uint8)
        split_image = np.ndarray((image_split_amount, split_size, split_size, reshaped_image.shape[2]),
                                 dtype=np.uint8)

        if labels is not None:
            for lay in range(image.shape[2]):
                reshaped_image[:, :, lay] = np.pad(image[:, :, lay], pads, mode='reflect')
                split_image[:, :, :, lay] = split_func(reshaped_image[:, :, lay])

            reshaped_labels = np.ndarray((new_shape[0], new_shape[1], labels.shape[2]), dtype=np.bool)
            split_labels = np.ndarray((image_split_amount, split_size, split_size, reshaped_labels.shape[2]),
                                      dtype=np.bool)
            for lay in range(labels.shape[2]):
                reshaped_labels[:, :, lay] = np.pad(labels[:, :, lay], pads, mode='reflect')
                split_labels[:, :, :, lay] = split_func(reshaped_labels[:, :, lay])

            return split_image, split_labels
        else:
           for lay in range(image.shape[2]):
                reshaped_image[:, :, lay] = np.pad(image[:, :, lay], pads, mode='reflect')
                split_image[:, :, :, lay] = split_func(reshaped_image[:, :, lay])

    else:
        reshaped_image = np.pad(np.squeeze(image), pads, mode='reflect')
        split_image = np.expand_dims(split_func(reshaped_image), axis=-1)

        if labels is not None:
            reshaped_labels = np.pad(np.squeeze(labels), pads, mode='reflect')
            split_labels = np.expand_dims(split_func(reshaped_labels), axis=-1)

            return split_image, split_labels

    return split_image


def merge(image, splited_images, overlap=OVERLAP):
    # 1. Calculate pads
    split_size = splited_images.shape[1]
    new_shape, images_amount = calculate_split_parameters(image.shape[0], image.shape[1], split_size, overlap)

    width_diff = new_shape[0] - image.shape[0]
    if width_diff % 2 == 0:
        left_pad = right_pad = int((new_shape[0] - image.shape[0]) / 2)
    else:
        left_pad = int((new_shape[0] - image.shape[0]) / 2)
        right_pad = int((new_shape[0] - image.shape[0]) / 2) + 1

    height_diff = new_shape[1] - image.shape[1]
    if height_diff % 2 == 0:
        top_pad = bottom_pad = int((new_shape[1] - image.shape[1]) / 2)
    else:
        top_pad = int((new_shape[1] - image.shape[1]) / 2)
        bottom_pad = int((new_shape[1] - image.shape[1]) / 2) + 1

    # 2. Merge splited images
    split_step = split_size - overlap
    merge_image_width = split_size + (split_step * (images_amount[0] - 1))
    merge_image_height = split_size + (split_step * (images_amount[1] - 1))

    def merge_func(images):
        image_merge_big = np.ndarray((merge_image_width, merge_image_height), dtype=np.float32)
        c = 0
        for i in range(0, merge_image_width - split_step, split_step):
            for j in range(0, merge_image_height - split_step, split_step):
                # print(image_merge[i:i + split_size, j:j + split_size].shape)
                image_merge_big[i:i + split_size, j:j + split_size] = images[c]
                c += 1
        # 3. Remove pads
        image_merge = image_merge_big[left_pad:-right_pad, top_pad:-bottom_pad]

        return image_merge

    if image.ndim == 3 and splited_images.shape[3] != 1:
        merge_image = np.ndarray((image.shape[0], image.shape[1], image.shape[2]), dtype=np.float32)
        for lay in range(image.shape[2]):
            merge_image = merge_func(splited_images[:, :, :, lay])
    elif image.ndim == 3 and splited_images.shape[3] == 1:
        merge_image = merge_func(splited_images[:, :, :, 0])
    else:
        merge_image = merge_func(splited_images)

    return merge_image


def rotate(image, labels=None, angle=0):
    cols, rows = image.shape[0], image.shape[1]
    M = cv2.getRotationMatrix2D((rows / 2, cols / 2), angle, 1)
    image_copy = image.copy()

    if image.ndim == 3:
        rotated_image = np.ndarray((image.shape[0], image.shape[1], image.shape[2]), dtype=np.uint8)
        rotated_labels = np.ndarray((labels.shape[0], labels.shape[1], labels.shape[2]), dtype=np.bool)

        for lay in range(image.shape[2]):
            rotated_image[:, :, lay] = cv2.warpAffine(image_copy[:, :, lay], M, (rows, cols))
            if labels is not None:
                labels_copy = labels.copy()
                rotated_labels[:, :, lay] = cv2.warpAffine(labels_copy[:, :, lay], M, (rows, cols))

                return rotated_image, rotated_labels
    else:
        rotated_image = cv2.warpAffine(image_copy, M, (rows, cols))
        if labels is not None:
            labels_copy = labels.copy()
            rotated_label = cv2.warpAffine(labels_copy, M, (rows, cols))

            return rotated_image, rotated_label

    return rotated_image


def rescale(image, labels=None, scale_factor=1):
    pass


"""
                    HISTOGRAM AUGMENTATIONS
                   (NORMALIZING, )
"""


def normalization(image, grid_size=8):

    def rgb_clahe(img):
        bgr = img[:, :, [2, 1, 0]]  # flip r and b
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(grid_size, grid_size))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        return bgr[:, :, [2, 1, 0]]

    def rgb_clahe_justl(img):
        bgr = img[:, :, [2, 1, 0]]  # flip r and b
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(grid_size, grid_size))
        return clahe.apply(lab[:, :, 0])

    # normalized_image = rgb_clahe(image)
    normalized_image = rgb_clahe_justl(image)

    # Invert if the intensity/background is too high
    mean = np.mean(normalized_image)
    # print(mean)
    if (mean > 127):
        normalized_image = (255-normalized_image)

    return normalized_image

def show_images(images):
    fig = plt.figure(1, figsize=(10, 10))
    for i, image in enumerate(images):
        ax = fig.add_subplot(1, 2, i+1)
        ax.imshow(np.squeeze(image))
    plt.show()


