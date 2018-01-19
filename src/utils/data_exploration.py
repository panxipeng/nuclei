import os, sys
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import random
import skimage.io
from collections import Counter
import matplotlib.pyplot as plt


TRAIN_PATH = '../data/images/train'
TEST_PATH = '../data/images/test'


# Get train and test IDs
def get_id():
    train_ids = next(os.walk(TRAIN_PATH))[1]
    test_ids = next(os.walk(TEST_PATH))[1]

    return train_ids, test_ids


def read_image_labels(image_id, img_type):
    # most of the content in this function is taken from 'Example Metric Implementation' kernel
    # by 'William Cukierski'
    if img_type == 'train':
        image_file = "../data/images/train/{}/images/{}.png".format(image_id,image_id)
        mask_file = "../data/images/train/{}/masks/*.png".format(image_id)
        image = skimage.io.imread(image_file)
        masks = skimage.io.imread_collection(mask_file).concatenate()
        height, width, _ = image.shape
        num_masks = masks.shape[0]
        labels = np.zeros((height, width), np.uint16)
        for index in range(0, num_masks):
            labels[masks[index] > 0] = index + 1
        return image, labels

    elif img_type == 'test':
        image_file = "../data/images/test/{}/images/{}.png".format(image_id, image_id)
        image = skimage.io.imread(image_file)
        labels = 0
        return image, labels


def plot_images_masks(image_ids):
    # plt.close('all')
    # fig, ax = plt.subplots(nrows=8,ncols=4, figsize=(15,15))

    sizes = []
    img_type = 'train'
    for ax_index, image_id in enumerate(image_ids):
        image, labels = read_image_labels(image_id, img_type)
        sizes.append('{}_{}'.format(image.shape[0], image.shape[1]))

        # img_row, img_col, mask_row, mask_col = int(ax_index/4)*2, ax_index%4, int(ax_index/4)*2 + 1, ax_index%4
        # ax[img_row][img_col].imshow(image)
        # ax[mask_row][mask_col].imshow(labels)
    # plt.show()
    c = Counter(sizes)
    print(c)


if __name__ == '__main__':
    train_ids, test_ids = get_id()

    print("Total Images in Training set: {}".format(len(train_ids)))
    # random_image_ids = random.sample(train_ids, 16)

    # print("Randomly Selected Images: {}, their IDs: {}".format(len(random_image_ids), random_image_ids))
    plot_images_masks(train_ids)
