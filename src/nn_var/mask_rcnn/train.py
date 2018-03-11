import os
import sys
import random
import math
import re
import time
import numpy as np
import datetime
from tqdm import tqdm
import skimage.io
import skimage.color
from sklearn.model_selection import train_test_split
import cv2
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
from src.nn_var.mask_rcnn.config import Config
from src.nn_var.mask_rcnn import utils
import src.nn_var.mask_rcnn.model as modellib
import src.nn_var.mask_rcnn.visualize as visualize
from src.nn_var.mask_rcnn.model import log
import src.nn_var.mask_rcnn.CellsDataset as CD
from dirs import ROOT_DIR, make_dir
from src.utils import data_exploration as de
from src.utils import encode_submit_for_mask_rcnn as esfmr
from src.utils import data_postprocessing as dpost
from src.utils import metric

# Dataset directory
DATASET_DIR = os.path.join(ROOT_DIR, r'data/fixed_images/train')
TEST_DATASET_DIR = os.path.join(ROOT_DIR, r'data/images/test')
VAL_DATASET_DIR = os.path.join(ROOT_DIR, r'data/images/validation')
# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, r'src/nn_var/mask_rcnn/logs')
# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, r'src/nn_var/mask_rcnn/coco_model/mask_rcnn_coco.h5')
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)


def get_model_name(params):
    name = params['model_type'] + '-'\
                 + "{}_ep-".format(params['epochs'])\
                 + "{}_vs-".format(params['val_split'])\
                 + "{}_iw-".format(params['init_with'])\
                 + "{}_l".format(params['layers'])\

    return name


def train_model(config, params, images_ids):
    """
    Train the model
    :param config: model configure class instance
    :param params: parameters dictionary
    :param model_name: the name of the model
    :param images_ids: train images ids
    :return:
    """
    # Prepare training and validation datasets
    val_split = params['val_split']
    train_images, val_images = de.split_test_val(source_ids=images_ids, val_split_factor=val_split)
    print("Train images: {}".format(len(train_images)) +
          "\nValidation images: {}".format(len(val_images)))

    dataset_train = CD.CellsDataset()
    dataset_train.load_cells(dataset_dir=DATASET_DIR, image_ids=train_images)
    dataset_train.prepare()
    dataset_val = CD.CellsDataset()
    dataset_val.load_cells(dataset_dir=DATASET_DIR, image_ids=val_images)
    dataset_val.prepare()

    # Create model object in "training" mode
    model = modellib.MaskRCNN(mode="training",
                              config=config,
                              model_dir=MODEL_DIR
                              )

    # Choose weights to start with
    init_with = params['init_with']
    if init_with == "imagenet":
        model.load_weights(model.get_imagenet_weights(), by_name=True)
    elif init_with == "coco":
        # Load weights trained on MS COCO, but skip layers that
        # are different due to the different number of classes
        # See README for instructions to download the COCO weights
        model.load_weights(COCO_MODEL_PATH, by_name=True,
                           exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                    "mrcnn_bbox", "mrcnn_mask"])
    elif init_with == "last":
        # Load the last model you trained and continue training
        model.load_weights(model.find_last()[1], by_name=True)
    else:
        model.load_weights(os.path.join(ROOT_DIR, params['path_to_weights_for_train']), by_name=True)
    print("Training starts with weights from {}".format(init_with))

    # Train the model
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=params['epochs'],
                layers=params['layers']
                )


def predict(config, params, model_name, images_ids):
    """
    Predict masks for test images
    :param config: model configure class instance
    :param params: parameters dictionary
    :param model_name: the name of the model
    :param images_ids: ids of images to predict (test_ids or val_ids)
    :return: path to the folder with predicted files
    """
    # Create model object in "inference" mode
    model = modellib.MaskRCNN(mode="inference",
                              config=config,
                              model_dir=MODEL_DIR)

    # Choose the weights for predicting
    model_path = ""
    if params['epoch_for_predict'].isdigit():
        model_path = os.path.join(model.find_last()[0], r'mask_rcnn_cell_00{}.h5'.format(params['epoch_for_predict']))
    elif params['epoch_for_predict'] == 'last':
        model_path = model.find_last()[1]
    elif params['epoch_for_predict'] == 'path':
        model_path = os.path.join(ROOT_DIR, params['path_to_weights_for_predict'])
    assert model_path != "", "Provide path to trained weights"
    print("Loading weights from ", model_path)
    model.load_weights(model_path, by_name=True)

    # Choose the type of predicting dataset:
    #   - val (validation) - predict validation dataset for model validation score estimation;
    #   - test - predict test dataset.
    predict_type = params['predict_type']
    pred_files_dir = ""
    images_dir = ""
    if predict_type == 'test':
        # Create folder for predicted files
        relative_preds_path = r'out_files\images\predict\{}-{}_pep'.format(model_name, params['epoch_for_predict'])
        pred_files_dir = make_dir(relative_preds_path)
        images_dir = TEST_DATASET_DIR
    elif predict_type == 'val':
        # Create folder for predicted files
        relative_preds_path = r'out_files\images\predict_val\{}-{}_pep'.format(model_name, params['epoch_for_predict'])
        pred_files_dir = make_dir(relative_preds_path)
        images_dir = DATASET_DIR
    assert pred_files_dir != "", "Provide path to predict files"
    assert images_dir != "", "Provide path to source files"

    # Save config file
    config.save_to_file(os.path.join(pred_files_dir, '{}.csv'.format(model_name)))

    # Predict images
    for image_id in tqdm(images_ids, total=len(images_ids)):
        # Create folder for image
        image_dir = os.path.join(pred_files_dir, image_id)
        if not os.path.exists(image_dir):
            os.mkdir(image_dir)

        # Read test image
        test_image = skimage.io.imread(os.path.join(images_dir, r'{}\images\{}.png'.format(image_id, image_id)))
        # If grayscale. Convert to RGB for consistency.
        if test_image.ndim != 3:
            test_image = skimage.color.gray2rgb(test_image)
        elif test_image.shape[2] > 3:
            test_image = test_image[:, :, :3]

        # Predict mask for the given image
        results = model.detect([test_image], verbose=0)
        r = results[0]

        # Save predicted masks
        for i in range(r['masks'].shape[2]):
            import warnings
            warnings.filterwarnings("ignore")
            r['masks'][:, :, i] *= 255
            skimage.io.imsave('{}\{}.png'.format(image_dir, i), r['masks'][:, :, i])

    return pred_files_dir


if __name__ == "__main__":
    # Make configs for training
    config = CD.CellsConfig()

    # Parameters dictionary
    params = {
        'model_type': 'mrcnn',
        'epochs': 60,
        'val_split': 0.2,
        'init_with': 'coco',  # imagenet, coco, last or path
        'path_to_weights_for_train': 'src/nn_var/mask_rcnn/logs/cell20180302T1515/mask_rcnn_cell_0026.h5',
        'layers': 'heads',  # heads or all
        'predict_type': 'val',  # test or val
        'epoch_for_predict': '26',
        'path_to_weights_for_predict': 'src/nn_var/mask_rcnn/logs/cell20180302T1515/mask_rcnn_cell_0026.h5'
    }

    # Make name for the model
    model_name = get_model_name(params=params)

    # Get images ids
    train_ids, test_ids = de.get_id()
    _, val_ids = de.split_test_val(train_ids, params['val_split'])

    # print('\n' + '-' * 30 + ' Training the model... ' + '-' * 30 + '\n')
    # config.save_to_file(os.path.join(MODEL_DIR, '{}.csv'.format(model_name)))
    # train_model(params=params, config=config, images_ids=train_ids)

    if params['predict_type'] == 'test':
        predict_ids = test_ids
        postproc_out_dir = r'out_files/images/postproc'
        print('\n' + '-' * 30 + ' Predicting test data... ' + '-' * 30 + '\n')
    elif params['predict_type'] == 'val':
        predict_ids = val_ids
        postproc_out_dir = r'out_files/images/postproc_val'
        print('\n' + '-' * 30 + ' Predicting validation data... ' + '-' * 30 + '\n')
    else:
        predict_ids = []
        postproc_out_dir = ''
        print("Predict datatype in parameters dict is incorrect")

    # Predict data
    predict_dir = predict(config=config, params=params, model_name=model_name, images_ids=predict_ids)

    # Data postprocessing
    print('\n' * 2 + '-' * 30 + ' Predicted data postprocessing... ' + '-' * 30 + '\n')
    postproc_model_name = predict_dir.replace("\\", "/").split("/")[-1]
    for image_id in tqdm(predict_ids, total=len(predict_ids)):
        labels = dpost.read_labels(predict_dir, image_id)
        morfling_labels = dpost.morfling(labels=labels)
        removed_instances_labels = dpost.remove_small_instances(labels=morfling_labels)
        overlap_fix_labels = dpost.overlapping_fix(labels=removed_instances_labels)
        dpost.save_labels(labels=overlap_fix_labels,
                          out_dir=postproc_out_dir,
                          process_type='remove_small_obj',
                          model_name=postproc_model_name,
                          image_id=image_id)

    # Check IoU score for validation data or encode submit for test data
    if params['predict_type'] == 'val':
        print('\n' * 2 + '-' * 30 + ' Check validation score... ' + '-' * 30 + '\n')
        pred_images_dir = r'out_files/images/postproc_val/remove_small_obj/{}'.format(postproc_model_name)
        val_ids = next(os.walk(os.path.join(ROOT_DIR, pred_images_dir)))[1]
        image_ids = val_ids
        true_images_dir = r'data/images/train'
        p = 0
        for image_id in tqdm(image_ids, total=len(image_ids)):
            p += metric.calculate_image_iou(image_id=image_id, true_images_dir=true_images_dir,
                                     pred_images_dir=pred_images_dir)

        mean_p = p / len(image_ids)
        print('\n\nTotal IoU for validation set: {:1.3f}'.format(mean_p))

    elif params['predict_type'] == 'test':
        print('\n' * 2 + '-' * 30 + ' Creating submit file... ' + '-' * 30 + '\n')
        images_to_encode = os.path.join(ROOT_DIR,
                                        r'out_files/images/postproc/remove_small_obj/{}'.format(postproc_model_name))

        submit_path = make_dir('sub/{}'.format(postproc_model_name))
        config.save_to_file(os.path.join(submit_path, 'config.csv'))
        esfmr.create_submit(files_path=images_to_encode, model_name=postproc_model_name, submit_path=submit_path)

