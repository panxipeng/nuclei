import numpy as np # linear algebra
import random
import skimage.io
import matplotlib.pyplot as plt
import cv2
import os
from dirs import ROOT_DIR

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


def show_image(image, labels=None):
    fig = plt.figure(1, figsize=(10, 10))

    if labels is None:
        fig.add_subplot(3, 2, 1).imshow(image)

        split_image = split(image)
        fig.add_subplot(3, 2, 3).imshow(split_image[5])

        rot_image = rotate(image, angle=90)
        fig.add_subplot(3, 2, 5).imshow(rot_image)

    else:
        fig.add_subplot(4, 2, 1).imshow(image)
        fig.add_subplot(4, 2, 2).imshow(labels[:, :, 0])

        split_image, split_labels = split(image, labels)
        fig.add_subplot(4, 2, 3).imshow(split_image[0, :, :, 0])
        fig.add_subplot(4, 2, 4).imshow(split_labels[0, :, :, 0])

        rot_image, rot_labels = rotate(image, labels, angle=90)
        fig.add_subplot(4, 2, 5).imshow(rot_image)
        fig.add_subplot(4, 2, 6).imshow(rot_labels[:, :, 0])

        normalized_image = normalization(image)
        fig.add_subplot(4, 2, 7).imshow(normalized_image)
        fig.add_subplot(4, 2, 8).imshow(labels[:, :, 0])

    plt.show()


def show_images(ids):
    plt_cols = len(ids) // 2

    fig = plt.figure(1, figsize=(10, 10))
    for i, id in enumerate(ids):
        ax = fig.add_subplot(plt_cols, 4, i*2 + 1)
        img, labels = read_train_image_labels(id)
        normalized_image = normalization(img)
        # print(normalized_image.shape)
        ax.imshow(normalized_image)

        split_image, split_labels = split(img, labels)
        # ax.imshow(split_image[6])

        merge_image = merge(img, split_image)
        ay = fig.add_subplot(plt_cols, 4, i*2 + 2)
        ay.imshow(merge_image)

    plt.show()


if __name__ == '__main__':
    # train_ids, test_ids = de.get_id()
    # trn_image, labels = read_train_image_labels(train_ids[0])
    # tst_image = read_test_image(test_ids[12])
    # split_tst = split(tst_image)
    # merge_tst = merge(tst_image, split_tst)

    X_train = np.load(os.path.join(ROOT_DIR,
    r'out_files/npy/predict/zf_turbo_unet_1-val_32-bs_100-ep_0.2-vs_splited-ts_128_128_X_test/0114f484a16c152baa2d82fdd43740880a762c93f436c8988ac461c5c9dbe7d5.npy'))
    Y_train = np.load(os.path.join(ROOT_DIR,
    r'out_files/npy/predict/zf_turbo_unet_1-val_32-bs_100-ep_0.2-vs_splited-ts_128_128_X_test/4be73d68f433869188fe5e7f09c7f681ed51003da6aa5d19ce368726d8e271ee.npy'))

    print(X_train.shape)

    random_image = random.randint(0, X_train.shape[0])
    fig = plt.figure(1, figsize=(10, 10))
    ax = fig.add_subplot(1, 2, 1)
    ax.imshow(np.squeeze(X_train))
    ay = fig.add_subplot(1, 2, 2)
    ay.imshow(np.squeeze(Y_train))
    plt.show()

    # show_image(tst_image)
    # show_image(trn_image, labels)
    # # rot_image, rot_labels = rotate(trn_image, labels, angle=90)
    #
    # random_image_ids = random.sample(train_ids, 8)
    # show_images(random_image_ids)

    # X_train = np.load(os.path.join(ROOT_DIR, r'out_files/npy/128_128_all_train/128_128_X_train.npy'))
    # Y_train = np.load(os.path.join(ROOT_DIR, r'out_files/npy/128_128_all_train/128_128_Y_train.npy'))
    #
    # print(X_train.shape)
    # print(Y_train.shape)
    #
    # random_image = random.randint(0, X_train.shape[0])
    #
    # fig = plt.figure(1, figsize=(10, 10))
    # ax = fig.add_subplot(1, 2, 1)
    # ax.imshow(X_train[random_image, :, :, 0])
    # ay = fig.add_subplot(1, 2, 2)
    # ay.imshow(Y_train[random_image, :, :, 0])
    # plt.show()

    # ids = []
    # for idx in range(Y_train.shape[0]):
    #     if np.mean(Y_train[idx]) == 0:
    #         ids.append(idx)
    #
    # print(ids)
    # new_X_train = np.delete(X_train, ids, axis=0)
    # new_Y_train = np.delete(Y_train, ids, axis=0)
    #
    # print(new_X_train.shape)
    # print(new_Y_train.shape)
    #
    # np.save(r'E:\Programming\ML Competitions\nuclei\out_files\npy\128_128_all_train\128_128_X_train_new.npy', new_X_train)
    # np.save(r'E:\Programming\ML Competitions\nuclei\out_files\npy\128_128_all_train\128_128_Y_train_new.npy', new_Y_train)

