import os
import numpy as np # linear algebra
import skimage.io
from collections import Counter
from dirs import ROOT_DIR
import matplotlib.pyplot as plt
import random

TRAIN_PATH = os.path.join(ROOT_DIR, r'data/images/train')
TEST_PATH = os.path.join(ROOT_DIR, r'data/images/test')


# Get train and test IDs
def get_id():
    train_ids = next(os.walk(TRAIN_PATH))[1]
    test_ids = next(os.walk(TEST_PATH))[1]

    # print(train_ids)
    return train_ids, test_ids


def read_image_labels(image_id, img_type):
    # most of the content in this function is taken from 'Example Metric Implementation' kernel
    # by 'William Cukierski'
    if img_type == 'train':
        image_file = os.path.join(ROOT_DIR, r'data/images/train/{}/images/{}.png'.format(image_id,image_id))
        mask_file = os.path.join(ROOT_DIR, r'data/images/train/{}/masks/*.png'.format(image_id))
        image = skimage.io.imread(image_file)
        masks = skimage.io.imread_collection(mask_file).concatenate()
        height, width, _ = image.shape
        num_masks = masks.shape[0]
        labels = np.zeros((height, width), np.uint16)
        for index in range(0, num_masks):
            labels[masks[index] > 0] = index + 1
        return image, labels

    elif img_type == 'test':
        image_file = os.path.join(ROOT_DIR, r'data/images/test/{}/images/{}.png'.format(image_id, image_id))
        image = skimage.io.imread(image_file)
        labels = 0
        return image, labels


def plot_images_masks(image_ids):
    # plt.close('all')
    # fig, ax = plt.subplots(nrows=8,ncols=4, figsize=(15,15))

    sizes = []
    img_type = 'test'
    for ax_index, image_id in enumerate(image_ids):
        image, labels = read_image_labels(image_id, img_type)
        sizes.append('{}_{}'.format(image.shape[0], image.shape[1]))

        # img_row, img_col, mask_row, mask_col = int(ax_index/4)*2, ax_index%4, int(ax_index/4)*2 + 1, ax_index%4
        # ax[img_row][img_col].imshow(image)
        # ax[mask_row][mask_col].imshow(labels)
    # plt.show()
    c = Counter(sizes)
    print(c)


def show_images(images):
    fig = plt.figure(1, figsize=(15, 15))
    for idx, image in enumerate(images):
        ax = fig.add_subplot(int(len(images)/4) + 1, 4, idx + 1)
        ax.imshow(np.squeeze(image))
    plt.show()


if __name__ == '__main__':
    tst_name = r'4f949bd8d914bbfa06f40d6a0e2b5b75c38bf53dbcbafc48c97f105bee4f8fac'
    preds_folder = r'zf_turbo_unet_gen-1_15-bs_50-ep_0.2-vs_split-ts_128_128_X_test'
    IMG_CHANNELS = 3

    test_image = skimage.io.imread(os.path.join(ROOT_DIR, r'data/images/test/{}/images/{}.png'.format(tst_name, tst_name)))[:, :, :IMG_CHANNELS]
    test_mask = np.load(os.path.join(ROOT_DIR, r'out_files/npy/predict/{}/{}.npy'.format(preds_folder, tst_name)))
    test_bin_mask = np.load(os.path.join(ROOT_DIR, r'out_files/npy/bin_predict/{}/{}.npy'.format(preds_folder, tst_name)))

    X_train = np.load(os.path.join(ROOT_DIR, r'out_files/npy/128_128_split/X_train.npy'))
    Y_train = np.load(os.path.join(ROOT_DIR, r'out_files/npy/128_128_split/Y_train.npy'))
    X_test = np.load(os.path.join(ROOT_DIR, r'out_files/npy/128_128_split/X_test.npy'))
    random_image = random.randint(0, X_train.shape[0])
    random_image_tst = random.randint(0, X_test.shape[0])

    # X_train = X_train.astype('float32')
    # Y_train = Y_train.astype('float32')

    # X_train /= 255.  # scale masks to [0, 1]

    X_train = X_train.astype('uint8')
    Y_train = Y_train.astype('uint8')

    images = [X_train[random_image], Y_train[random_image], X_test[random_image_tst]]
    print(np.max(test_image), np.min(test_image))
    print(np.max(test_mask), np.min(test_mask))
    print(np.max(test_bin_mask), np.min(test_bin_mask))
    show_images(images)

