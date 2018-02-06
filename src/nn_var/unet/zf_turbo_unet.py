# coding: utf-8

from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Dropout, Activation
from keras.layers.merge import concatenate
import numpy as np
from keras.models import Model
from keras.layers import Input
from keras.layers.core import Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras import backend as K
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam, Nadam, SGD
import tensorflow as tf
from keras.losses import binary_crossentropy

class ZF_Unet():
    def __init__(self, img_rows, img_cols, img_channels):
        self.model_type = 'zf_turbo_unet'
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.img_channels = img_channels

    def mean_iou(self, y_true, y_pred):
        prec = []
        for t in np.arange(0.5, 1.0, 0.05):
            y_pred_ = tf.to_int32(y_pred > t)
            score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
            K.get_session().run(tf.local_variables_initializer())
            with tf.control_dependencies([up_opt]):
                score = tf.identity(score)
            prec.append(score)

        return K.mean(K.stack(prec), axis=0)

    def dice_coef(self, y_true, y_pred):
        smooth = 1.
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

    def bce_dice_loss(self, y_true, y_pred):
        return 0.5 * binary_crossentropy(y_true, y_pred) - self.dice_coef(y_true, y_pred)

    def double_conv_layer(self, x, size, dropout, batch_norm):
        axis = 3
        conv = Conv2D(size, (3, 3), activation='relu', padding='same')(x)
        if batch_norm is True:
            conv = BatchNormalization(axis=axis)(conv)
        # conv = Activation('relu')(conv)
        conv = Conv2D(size, (3, 3), activation='relu', padding='same')(conv)
        if batch_norm is True:
            conv = BatchNormalization(axis=axis)(conv)
        # conv = Activation('relu')(conv)
        if dropout > 0:
            conv = Dropout(dropout)(conv)
        return conv

    def get_unet(self, filters=4, dropout_val=0.2, batch_norm=True):
        inputs = Input((self.img_rows, self.img_cols, self.img_channels))

        bn = BatchNormalization()(inputs)
        conv_224 = self.double_conv_layer(bn, filters, dropout_val, batch_norm)
        pool_112 = MaxPooling2D(pool_size=(2, 2))(conv_224)

        conv_112 = self.double_conv_layer(pool_112, 2 * filters, dropout_val, batch_norm)
        pool_56 = MaxPooling2D(pool_size=(2, 2))(conv_112)

        conv_56 = self.double_conv_layer(pool_56, 4 * filters, dropout_val, batch_norm)
        pool_28 = MaxPooling2D(pool_size=(2, 2))(conv_56)

        conv_28 = self.double_conv_layer(pool_28, 8 * filters, dropout_val, batch_norm)
        pool_14 = MaxPooling2D(pool_size=(2, 2))(conv_28)

        conv_14 = self.double_conv_layer(pool_14, 16 * filters, dropout_val, batch_norm)
        pool_7 = MaxPooling2D(pool_size=(2, 2))(conv_14)

        conv_7 = self.double_conv_layer(pool_7, 32 * filters, dropout_val, batch_norm)

        up_14 = concatenate([UpSampling2D(size=(2, 2))(conv_7), conv_14], axis=-1)
        up_conv_14 = self.double_conv_layer(up_14, 16 * filters, dropout_val, batch_norm)

        up_28 = concatenate([UpSampling2D(size=(2, 2))(up_conv_14), conv_28], axis=-1)
        up_conv_28 = self.double_conv_layer(up_28, 8 * filters, dropout_val, batch_norm)

        up_56 = concatenate([UpSampling2D(size=(2, 2))(up_conv_28), conv_56], axis=-1)
        up_conv_56 = self.double_conv_layer(up_56, 4 * filters, dropout_val, batch_norm)

        up_112 = concatenate([UpSampling2D(size=(2, 2))(up_conv_56), conv_112], axis=-1)
        up_conv_112 = self.double_conv_layer(up_112, 2 * filters, dropout_val, batch_norm)

        up_224 = concatenate([UpSampling2D(size=(2, 2))(up_conv_112), conv_224], axis=-1)
        up_conv_224 = self.double_conv_layer(up_224, filters, 0, batch_norm)

        conv_final = Conv2D(1, (1, 1))(up_conv_224)
        # conv_final = BatchNormalization(axis=-1)(conv_final)
        conv_final = Activation('sigmoid')(conv_final)

        model = Model(inputs, conv_final)

        adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        nadam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

        model.compile(optimizer=adam, loss=self.bce_dice_loss, metrics=[self.mean_iou])

        return model
