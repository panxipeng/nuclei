import os
import numpy as np
from dirs import ROOT_DIR
from keras.models import model_from_json


def predict_submit(model_name, file_to_predict, predict_type, params):
    print('-' * 30 + ' Loading model... ' + '-' * 30)
    json_file = open(os.path.join(ROOT_DIR, r'models') + '/' + model_name + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights(os.path.join(ROOT_DIR, r'models') + '/' + model_name + '.h5')

    print('-' * 30 + ' Predicting masks on data... ' + '-' * 30)
    predict_file = model.predict(file_to_predict, verbose=2)

    predict_img_path = os.path.join(ROOT_DIR, r'out_files\predict\{}_{}_{}'
                                    .format(model_name, params['img_rows'], params['img_cols']))
    if not os.path.exists(predict_img_path):
        os.mkdir(predict_img_path)

    print('-' * 30 + ' Save predicted file as npy... ' + '-' * 30)
    predict_file_out_name = predict_img_path + '/' + predict_type + '.npy'
    np.save(predict_file_out_name, predict_file)

    return predict_file

