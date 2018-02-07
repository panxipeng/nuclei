import numpy as np # linear algebra
from scipy import signal
import random
import skimage.io
import matplotlib.pyplot as plt
import cv2
import os
from dirs import ROOT_DIR
from PIL import Image
from pylab import *


def contours(image_id):
    # img = cv2.imread(os.path.join(ROOT_DIR, r'data/images/test/{}/images/{}.png'.format(image_id, image_id)), 0)
    img = cv2.imread(r'E:\Programming\ML Competitions\nuclei\data\images\train\0a7d30b252359a10fd298b638b90cb9ada3acced4e0c0e5a3692013f432ee4e9\masks\0adbf56cd182f784ca681396edc8b847b888b34762d48168c7812c79d145aa07.png', 0)
    # img = np.load(os.path.join(ROOT_DIR, r'out_files/npy/128_128_split/Y_train.npy'))[0]
    # img = np.squeeze(img)
    print(type(img))
    print(img.shape)
    edges = cv2.Canny(img, 0, 100)
    im2, contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contour_img = edges

    return contour_img


def show_images(images):
    fig = plt.figure(1, figsize=(10, 10))
    for i, image in enumerate(images):
        ax = fig.add_subplot(1, 2, i+1)
        ax.imshow(image)
    plt.show()

def contours_detecting(image):
    gradient_magnitude_image = np.ndarray((image.shape[0], image.shape[1], image.shape[2]), dtype=float)
    gradmag = image.copy()
    xder_flt = np.asarray([[-1, 0, 1],
                           [-2, 0, 2],
                           [-1, 0, 1]])
    yder_flt = np.asarray([[1, 2, 1],
                           [0, 0, 0],
                           [-1, -2, -1]])

    for lay in range(gradmag.shape[2]):
        img = gradmag[:, :, lay]
        xder_reshape = np.pad(img, 1, mode='reflect')
        xder = signal.convolve2d(xder_reshape, xder_flt, mode='valid')
        yder_reshape = np.pad(img, 1, mode='reflect')
        yder = signal.convolve2d(yder_reshape, yder_flt, mode='valid')
        gradmag_out = np.hypot(xder, yder)
        gradient_magnitude_image[:, :, lay] = gradmag_out

    return gradient_magnitude_image


if __name__ == "__main__":
    images = np.load(os.path.join(ROOT_DIR, r'out_files/npy/128_128_split/X_train.npy'))

    image_id = r'9f17aea854db13015d19b34cb2022cfdeda44133323fcd6bb3545f7b9404d8ab'
    img = cv2.imread(os.path.join(ROOT_DIR, r'data/images/test/{}/images/{}.png'.format(image_id, image_id)), 0)
    cnt = contours(image_id)
    images = [cnt, img]
    show_images(images)
    # show_image(images[0])