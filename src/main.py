import os
import numpy as np
from src.nn_var.unet.zf_turbo_unet import ZF_Unet

# from src.utils import fit_save_without_gen as fs
from src.utils import fit_save_with_gen as fswg
from src.utils import predict_submit as ps
from src.utils import data_exploration as de

from src.utils import encode_submit as es
from dirs import ROOT_DIR

params = {
          'img_rows': 128,
          'img_cols': 128,
          'img_channels': 3,
          'msk_channels': 1,
          'batch_size': 32,
          'val_split': 0.2,
          'epochs': 100,
          'verbose': 2,
          'train_set': 'split',
          'gen_use': 1
          }


def get_model_name(model_type, params):
    name = str(model_type)\
                 + "_gen-{}_".format(params['gen_use'])\
                 + "{}-bs_".format(params['batch_size']) \
                 + "{}-ep_".format(params['epochs'])\
                 + "{}-vs_".format(params['val_split'])\
                 + "{}-ts".format(params['train_set'])

    return name


if __name__ == '__main__':
    # Get model
    model_class = ZF_Unet(img_rows=params['img_rows'],
                       img_cols=params['img_cols'],
                       img_channels=params['img_channels'])

    model = model_class.get_unet()

    # Get model name
    model_name = get_model_name(model_type=model_class.model_type, params=params)

    # Fitting model
    X_train = np.load(os.path.join(ROOT_DIR, r'out_files/npy/128_128_split/X_train.npy'))
    Y_train = np.load(os.path.join(ROOT_DIR, r'out_files/npy/128_128_split/Y_train.npy'))

    X_train = X_train.astype('uint8')
    Y_train = Y_train.astype('uint8')

    # X_train /= 255

    # fs.fit_save(model=model, model_name=model_name, X_train=X_train, Y_train=Y_train, params=params)
    # fswg.fit_save(model=model, model_name=model_name, X_train=X_train, Y_train=Y_train, params=params)

    # Predicting
    predict_type = "X_test"
    bin_predict_npy_path = ps.predict_submit(model_name=model_name, predict_type=predict_type, params=params)

    # Make submit
    train_ids, test_ids = de.get_id()
    print('-' * 30 + ' Creating submit file... ' + '-' * 30)
    es.create_submit(test_ids=test_ids, predict_files_path=bin_predict_npy_path, model_name=model_name)
