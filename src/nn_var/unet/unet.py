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

IMG_ROWS = 128
IMG_COLS = 128
IMG_CHANNELS = 3
MSK_CHANNELS = 1

BATCH_SIZE = 32
VALIDATION_SPLIT = 0.1
EPOCHS = 100
VERBOSE = 2


# Define IoU metric
def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)

    return K.mean(K.stack(prec), axis=0)


def get_unet():
    inputs = Input((IMG_ROWS, IMG_COLS, IMG_CHANNELS))
    s = Lambda(lambda x: x / 255)(inputs)

    # conv1 = BatchNormalization()(inputs)
    conv1 = Conv2D(8, (3, 3), activation='relu', padding='same')(s)
    conv1 = Conv2D(8, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(16, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(16, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(32, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv5)

    up6 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    conv6 = Conv2D(64, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = Conv2D(32, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(16, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(16, (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([Conv2DTranspose(8, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(8, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(8, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])

    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=[mean_iou])

    return model


def save_model(model, model_type, batch_size, nb_epoch):
    model_json = model.to_json()
    model_name = model_type + "_" + batch_size + "batch_" + nb_epoch + "epoch"
    json_file = open(os.path.join(ROOT_DIR, 'src/nn_var/unet') + '/' + model_name + ".json", "w")
    json_file.write(model_json)
    json_file.close()
    model.save_weights(os.path.join(ROOT_DIR, 'src/nn_var/unet') + '/' + model_name + ".h5")

    return model_name


def get_callbacks(filepath, patience=2):
   earlyStopping = EarlyStopping(patience=patience, verbose=2)
   msave = ModelCheckpoint(filepath, verbose=2, save_best_only=True)

   return [earlyStopping, msave]


def fit_save(X_train, Y_train):
    model = get_unet()
    gen = ImageDataGenerator(horizontal_flip=True,
                             vertical_flip=True,
                             width_shift_range=0,
                             height_shift_range=0,
                             channel_shift_range=0,
                             zoom_range=0.2,
                             shear_range=0.2,
                             rotation_range=20)

    # earlyStopping = EarlyStopping(patience=10, verbose=2)
    # msave = ModelCheckpoint('model-dsbowl2018-1.h5', verbose=2, save_best_only=True)
    callbacks = get_callbacks(filepath='model-dsbowl2018-1.h5', patience=10)

    # model.fit_generator(gen.flow(X_train, Y_train, batch_size=BATCH_SIZE, seed=55),
    #                           steps_per_epoch=50,
    #                           epochs=EPOCHS,
    #                           shuffle=True,
    #                           verbose=VERBOSE,
    #                           callbacks=[earlyStopping, msave])
    model.fit(X_train, Y_train,
              validation_split=VALIDATION_SPLIT,
              batch_size=BATCH_SIZE,
              epochs=EPOCHS,
              verbose=VERBOSE,
              callbacks=callbacks)

    print('-' * 30)
    print('Saving model...')
    print('-' * 30)
    model_name = save_model(model, model_type="unet", batch_size=str(BATCH_SIZE), nb_epoh=str(EPOCHS))

    return model_name


def predict(model_name, file_to_predict, predict_type):
    predict_img_path = os.path.join(ROOT_DIR,
                                    r'out_files\predict\{}_{}_{}'.format(model_name, IMG_ROWS, IMG_COLS))
    if not os.path.exists(predict_img_path):
        os.mkdir(predict_img_path)

    print('-' * 30)
    print('Loading model...')
    print('-' * 30)
    json_file = open(os.path.join(ROOT_DIR, 'src/nn_var/unet') + '/' + model_name + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights(os.path.join(ROOT_DIR, 'src/nn_var/unet') + '/' + model_name + '.h5')

    print('-' * 30)
    print('Loading and preprocessing data to predict...')
    print('-' * 30)


    print('-' * 30)
    print('Predicting masks on data...')
    print('-' * 30)
    predict_file = model.predict(file_to_predict, verbose=2)

    predict_file_out_name = predict_img_path + '/' + predict_type + '.npy'
    np.save(predict_file_out_name, predict_file)


if __name__ == '__main__':
    X_train = np.load(os.path.join(ROOT_DIR, r'out_files/npy/{}_{}/X_train.npy'.format(IMG_ROWS, IMG_COLS)))
    Y_train = np.load(os.path.join(ROOT_DIR, r'out_files/npy/{}_{}/Y_train.npy'.format(IMG_ROWS, IMG_COLS)))

    model_name = fit_save(X_train, Y_train)

    X_test = np.load(os.path.join(ROOT_DIR, r'out_files/npy/{}_{}/X_test.npy'.format(IMG_ROWS, IMG_COLS)))
    predict_type = 'X_test'
    predict(model_name, X_test, predict_type)

    # predict(model_name, X_train, 'X_train')

