import os
import numpy as np
import random

from src.nn_var.unet.unet import Unet
from src.utils import fit_save as fs
from src.utils import predict_submit as ps
from src.utils import starting_point as sp
from src.utils import encode_submit as es
from dirs import ROOT_DIR


seed = 42
random.seed = seed
np.random.seed = seed

params = {
          'img_rows': 128,
          'img_cols': 128,
          'img_channels': 3,
          'msk_channels': 1,
          'batch_size': 32,
          'val_split': 0.1,
          'epochs': 200,
          'verbose': 2
          }

def get_model_name(model_type, batch_size, nb_epoch, val_split):
    model_name = str(model_type)\
                 + "_"\
                 + str(batch_size)\
                 + "batch_" \
                 + str(nb_epoch)\
                 + "epoch_"\
                 + str(val_split)\
                 + "val-split"

    return model_name

if __name__ == '__main__':
    model_class = Unet(img_rows=params['img_rows'],
                       img_cols=params['img_cols'],
                       img_channels=params['img_channels'])

    model = model_class.get_unet()

    model_name = get_model_name(model_type=model_class.model_type,
                                batch_size=params['batch_size'],
                                nb_epoch=params['epochs'],
                                val_split=params['val_split'])

    X_train = np.load(os.path.join(ROOT_DIR, r'out_files/npy/{}_{}/X_train.npy'
                                   .format(params['img_rows'], params['img_cols'])))
    Y_train = np.load(os.path.join(ROOT_DIR, r'out_files/npy/{}_{}/Y_train.npy'
                                   .format(params['img_rows'], params['img_cols'])))

    fs.fit_save(model=model,
                model_name=model_name,
                X_train=X_train,
                Y_train=Y_train,
                params=params)

    X_test = np.load(os.path.join(ROOT_DIR, r'out_files/npy/{}_{}/X_test.npy'
                                  .format(params['img_rows'], params['img_cols'])))
    train_ids, test_ids = sp.get_id()
    predict_type = "X_test"

    predict_file = ps.predict_submit(model_name=model_name,
                                     file_to_predict=X_test,
                                     predict_type=predict_type,
                                     params=params)

    print('-' * 30 + ' Creating submit file... ' + '-' * 30)
    es.create_submit(test_ids=test_ids,
                     predict_file=predict_file,
                     model_name=model_name)