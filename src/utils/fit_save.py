import numpy as np
import os

from keras.models import Model, model_from_json
from keras.layers import Input
from keras.layers.core import Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras import backend as K
from keras.optimizers import Adam, Nadam, SGD
import threading
import random

import tensorflow as tf
from dirs import ROOT_DIR
from src.nn_var.unet import unet
from src.encode_submit import create_submit
from src.starting_point import get_id

MODEL = unet.get_unet()

IMG_ROWS = 128
IMG_COLS = 128
IMG_CHANNELS = 3
MSK_CHANNELS = 1

BATCH_SIZE = 32
VALIDATION_SPLIT = 0.1
EPOCHS = 1
VERBOSE = 2

seed = 42
random.seed = seed
np.random.seed = seed


"""
UTIL METHODS
"""
def save_model(model, model_type, batch_size, nb_epoch):
    model_json = model.to_json()
    model_name = model_type + "_" + batch_size + "batch_" + nb_epoch + "epoch"
    json_file = open(os.path.join(ROOT_DIR, 'src/nn_var/unet') + '/' + model_name + ".json", "w")
    json_file.write(model_json)
    json_file.close()
    model.save_weights(os.path.join(ROOT_DIR, 'src/nn_var/unet') + '/' + model_name + ".h5")

    return model_name

def get_callbacks(filepath, patience=2):
   earlyStopping = EarlyStopping(patience=patience, verbose=VERBOSE)
   msave = ModelCheckpoint(filepath, verbose=VERBOSE, save_best_only=True)

   return [earlyStopping, msave]


"""
FITTING MODEL
"""
def fit_save(X_train, Y_train):
    model = MODEL

    callbacks = get_callbacks(filepath='model-dsbowl2018-1.h5', patience=10)

    model.fit(X_train, Y_train,
              validation_split=VALIDATION_SPLIT,
              batch_size=BATCH_SIZE,
              epochs=EPOCHS,
              verbose=VERBOSE,
              callbacks=callbacks)

    print('-' * 30)
    print('Saving model...')
    print('-' * 30)
    model_name = save_model(model, model_type="unet", batch_size=str(BATCH_SIZE), nb_epoch=str(EPOCHS))

    return model_name
