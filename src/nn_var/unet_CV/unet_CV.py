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
from sklearn.model_selection import StratifiedKFold
from keras import backend as K

import tensorflow as tf
from dirs import ROOT_DIR

IMG_ROWS = 128
IMG_COLS = 128
IMG_CHANNELS = 3

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

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[mean_iou])

    return model


def get_callbacks(filepath, patience=2):
   earlyStopping = EarlyStopping(patience=10, verbose=2, mode='min')
   msave = ModelCheckpoint(filepath, save_best_only=True)
   reduce_lr_loss = ReduceLROnPlateau(factor=0.1,
                                      patience=patience-3,
                                      verbose=2)

   return [earlyStopping, msave, reduce_lr_loss]


def unet_CV(gen, X_train, X_test, Y_train):
    K = 2

    # print(Y_train.shape)
    nsamples, nx, ny, nz = Y_train.shape
    d2_Y_train = Y_train.reshape((nsamples, nx * ny* nz))

    folds = list(StratifiedKFold(n_splits=K, shuffle=True, random_state=16).split(X_train, d2_Y_train))
    batch_size = 64
    epochs = 100
    verbose = 2
    steps_per_epoch = 240

    y_test_pred_log = 0
    y_train_pred_log = 0
    y_valid_pred_log = 0.0 * Y_train
    for j, (train_idx, test_idx) in enumerate(folds):
        print('\n===================FOLD=', j)
        X_train_cv = X_train[train_idx]
        y_train_cv = Y_train[train_idx]
        X_holdout = X_train[test_idx]
        Y_holdout = Y_train[test_idx]

        file_path = "%s_aug_basic_cnn_model_weights.hdf5" % j
        callbacks = get_callbacks(filepath=file_path, patience=10)
        basic_model = get_unet()
        basic_model.fit_generator(gen.flow(X_train_cv, y_train_cv, batch_size=batch_size, seed=55),
                                  steps_per_epoch=steps_per_epoch,
                                  epochs=epochs,
                                  shuffle=True,
                                  verbose=verbose,
                                  validation_data=(X_holdout, Y_holdout),
                                  callbacks=callbacks)
        basic_model.load_weights(filepath=file_path)
        # Getting Training Score
        score = basic_model.evaluate(X_train_cv, y_train_cv, verbose=verbose)
        print('Train loss:', score[0])
        print('Train accuracy:', score[1])
        # Getting Test Score
        score = basic_model.evaluate(X_holdout, Y_holdout, verbose=verbose)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])

        # Getting validation Score.
        pred_valid = basic_model.predict(X_holdout)
        y_valid_pred_log[test_idx] = pred_valid.reshape(pred_valid.shape[0])

        # Getting Test Scores
        temp_test = basic_model.predict(X_test)
        y_test_pred_log += temp_test.reshape(temp_test.shape[0])

        # Getting Train Scores
        temp_train = basic_model.predict(X_train)
        y_train_pred_log += temp_train.reshape(temp_train.shape[0])

    y_test_pred_log = y_test_pred_log / K
    y_train_pred_log = y_train_pred_log / K

    # print('\n Train Log Loss Validation= ', log_loss(Y_train, y_train_pred_log))
    # print(' Test Log Loss Validation= ', log_loss(Y_train, y_valid_pred_log))

    return y_test_pred_log #, log_loss(Y_train, y_valid_pred_log)


def load_and_predict(model_name, predict_type):
    gen = ImageDataGenerator(horizontal_flip=True,
                             vertical_flip=True,
                             width_shift_range=0,
                             height_shift_range=0,
                             channel_shift_range=0,
                             zoom_range=0.2,
                             shear_range=0.2,
                             rotation_range=20)
    X_train = np.load(os.path.join(ROOT_DIR, r'out_files/npy/default/X_train.npy'))
    Y_train = np.load(os.path.join(ROOT_DIR, r'out_files/npy/default/Y_train.npy'))
    X_test = np.load(os.path.join(ROOT_DIR, r'out_files/npy/default/X_test.npy'))

    preds = unet_CV(gen, X_train, X_test, Y_train)

    # model_parameters = {
    #     "model_type": "basic_cnn",
    #     "train_type": 'norm',
    #     "test_type": 'norm'
    #     # "val_score": "{0:.4f}".format(score)
    # }

    predict_img_path = os.path.join(ROOT_DIR,
                                    r'out_files\predict\{}_{}_{}'.format(model_name, IMG_ROWS, IMG_COLS))
    if not os.path.exists(predict_img_path):
        os.mkdir(predict_img_path)

    predict_file_out_name = predict_img_path + '/' + predict_type + '.npy'
    np.save(predict_file_out_name, preds)

    # return preds #, model_parameters
#
#
# def save_model(model, model_type, batch_size, nb_epoh):
#     model_json = model.to_json()
#     model_name = model_type + "_" + batch_size + "batch_" + nb_epoh + "epoch"
#     json_file = open(os.path.join(ROOT_DIR, 'src/nn_var/unet') + '/' + model_name + ".json", "w")
#     json_file.write(model_json)
#     json_file.close()
#     model.save_weights(os.path.join(ROOT_DIR, 'src/nn_var/unet') + '/' + model_name + ".h5")
#
#     return model_name
#
# def fit_save(X_train, Y_train):
#     model = get_unet()
#
#     earlystopper = EarlyStopping(patience=5, verbose=2)
#     checkpointer = ModelCheckpoint('model-dsbowl2018-1.h5', verbose=2, save_best_only=True)
#
#     batch_size = 32
#     validation_split = 0.1
#     epoch = 100
#     verbose = 2
#     # print(ROOT_DIR)
#
#     model.fit(X_train, Y_train,
#               validation_split=validation_split,
#               batch_size=batch_size,
#               epochs=epoch,
#               verbose=verbose,
#               callbacks=[earlystopper, checkpointer])
#
#     print('-' * 30)
#     print('Saving model...')
#     print('-' * 30)
#     model_name = save_model(model, model_type="unet", batch_size=str(batch_size), nb_epoh=str(epoch))
#
#     return model_name
#
#
# def predict(model_name, file_to_predict, predict_type):
#     predict_img_path = os.path.join(ROOT_DIR,
#                                     r'out_files\predict\{}_{}_{}'.format(model_name, IMG_ROWS, IMG_COLS))
#     if not os.path.exists(predict_img_path):
#         os.mkdir(predict_img_path)
#
#     print('-' * 30)
#     print('Loading model...')
#     print('-' * 30)
#     json_file = open(os.path.join(ROOT_DIR, 'src/nn_var/unet') + '/' + model_name + '.json', 'r')
#     loaded_model_json = json_file.read()
#     json_file.close()
#     model = model_from_json(loaded_model_json)
#     model.load_weights(os.path.join(ROOT_DIR, 'src/nn_var/unet') + '/' + model_name + '.h5')
#
#     print('-' * 30)
#     print('Loading and preprocessing data to predict...')
#     print('-' * 30)
#
#
#     print('-' * 30)
#     print('Predicting masks on data...')
#     print('-' * 30)
#     predict_file = model.predict(file_to_predict, verbose=2)
#
#     predict_file_out_name = predict_img_path + '/' + predict_type + '.npy'
#     np.save(predict_file_out_name, predict_file)


if __name__ == '__main__':
    model_name = 'unetCV_32batch_100epoch'
    predict_type = 'X_test'
    load_and_predict(model_name, predict_type)

    # predict(model_name, X_train, 'X_train')

