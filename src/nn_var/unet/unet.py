import os
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


class Unet():
    def __init__(self, img_rows, img_cols, img_channels):
        self.model_type = 'unet'
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

    def get_unet(self):
        inputs = Input((self.img_rows, self.img_cols, self.img_channels))
        # s = Lambda(lambda x: x / 255)(inputs)

        conv1 = BatchNormalization()(inputs)
        conv1 = Conv2D(8, (3, 3), activation='relu', padding='same')(conv1)
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

        model.compile(optimizer=adam, loss='binary_crossentropy', metrics=[self.mean_iou])

        return model

if __name__ == '__main__':
    pass


