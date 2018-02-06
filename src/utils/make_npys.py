import numpy as np
import os

from dirs import ROOT_DIR
from src.utils import data_exploration as de
from src.utils import data_augmentation as da


def make_train_npy(train_ids, save_dir):
    images_to_concatenate = []
    labels_to_concatenate = []
    for id in train_ids:
        image, labels = da.read_train_image_labels(id)
        # normalized_image = da.normalization(image)
        split_image, split_labels = da.split(image, labels)
        images_to_concatenate.append(split_image)
        labels_to_concatenate.append(split_labels)

    splited_train_images = np.concatenate(images_to_concatenate, axis=0)
    splited_train_labels = np.concatenate(labels_to_concatenate, axis=0)

    path = os.path.join(save_dir, r'{}_{}_split'.format(da.SPLIT_SIZE, da.SPLIT_SIZE))
    if not os.path.exists(path):
        os.mkdir(path)

    np.save(os.path.join(path, r'X_train.npy'), splited_train_images)
    np.save(os.path.join(path, r'Y_train.npy'), splited_train_labels)


def make_test_npy(test_ids, save_dir):
    images_to_concatenate = []
    for id in test_ids:
        image = da.read_test_image(id)
        split_image = da.split(image)
        images_to_concatenate.append(split_image)

    splited_test_images = np.concatenate(images_to_concatenate, axis=0)

    path = os.path.join(save_dir, r'{}_{}_split'.format(da.SPLIT_SIZE, da.SPLIT_SIZE))
    if not os.path.exists(path):
        os.mkdir(path)

    np.save(os.path.join(path, r'X_test.npy'), splited_test_images)


if __name__ == '__main__':
    train_ids, test_ids = de.get_id()
    save_dir = os.path.join(ROOT_DIR, r'out_files/npy')
    make_train_npy(train_ids, save_dir)
    # make_test_npy(test_ids, save_dir)


