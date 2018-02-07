import os
import numpy as np
from dirs import ROOT_DIR
from keras.models import model_from_json
from src.utils import data_preprocessing as dp
from src.utils import data_postprocessing as dpost

from src.utils import data_augmentation as da


def predict(model_name, predict_type, params):
    """
    Predicting images mask

    :param model_name: The name of the model
    :param predict_type: Type of the predicting images (train or test)
    :param params: Parameters dictionary
    :return: Path to the predicted images in npy-format
    """

    # Step 1: loading model from ../models
    print('-' * 30 + ' Loading model... ' + '-' * 30)
    json_file = open(os.path.join(ROOT_DIR, r'models') + '/' + model_name + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights(os.path.join(ROOT_DIR, r'models') + '/' + model_name + '.h5')

    # Step 2: Predicting masks
    print('-' * 30 + ' Predicting masks on data... ' + '-' * 30)
    predict_npy_path = os.path.join(ROOT_DIR, r'out_files/npy/predict/{}_{}_{}_{}'
                                    .format(model_name, params['img_rows'], params['img_cols'], predict_type))
    bin_predict_npy_path = os.path.join(ROOT_DIR, r'out_files/npy/bin_predict/{}_{}_{}_{}'
                                    .format(model_name, params['img_rows'], params['img_cols'], predict_type))
    if not os.path.exists(predict_npy_path):
        os.mkdir(predict_npy_path)
    if not os.path.exists(bin_predict_npy_path):
        os.mkdir(bin_predict_npy_path)

    train_ids, test_ids = dp.get_id()

    for i, ids in enumerate(test_ids):
        test_image = da.read_test_image(ids)
        splited_test_images = da.split(test_image)
        predict_images = model.predict(splited_test_images, verbose=2)
        predict_image = da.merge(test_image, predict_images)
        bin_predict_image = dpost.binarize(predict_image)
        predict_file_out_name = os.path.join(predict_npy_path, r'{}.npy'.format(ids))
        np.save(predict_file_out_name, predict_image)
        predict_file_out_name = os.path.join(bin_predict_npy_path, r'{}.npy'.format(ids))
        np.save(predict_file_out_name, bin_predict_image)

    return bin_predict_npy_path


if __name__ == '__main__':
    pass

