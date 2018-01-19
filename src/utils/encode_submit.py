import os
import numpy as np
import sys
from tqdm import tqdm
import pandas as pd
from skimage.io import imread
from skimage.transform import resize
from skimage.morphology import label

from src.starting_point import get_id, TEST_PATH
from dirs import ROOT_DIR

IMG_CHANNELS = 3

def predict_proc(test_ids, predict_file):
    sizes_test = []
    sys.stdout.flush()
    for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
        path = os.path.join(TEST_PATH, id_)
        img = imread(path + '/images/' + id_ + '.png')[:, :, :IMG_CHANNELS]
        sizes_test.append([img.shape[0], img.shape[1]])

    # Threshold predictions
    preds_test_upsampled = []
    for i in range(len(predict_file)):
        preds_test_upsampled.append(resize(np.squeeze(predict_file[i]),
                                           (sizes_test[i][0], sizes_test[i][1]),
                                           mode='constant', preserve_range=True))

    return preds_test_upsampled


def rle_encoding(x):
    dots = np.where(x.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b > prev + 1): run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths


def prob_to_rles(x, cutoff=0.5):
    lab_img = label(x > cutoff)
    for i in range(1, lab_img.max() + 1):
        yield rle_encoding(lab_img == i)


def create_submit(test_ids, predict_file, model_name):
    submit_path = os.path.join(ROOT_DIR, r'out_files\sub')
    if not os.path.exists(submit_path):
        os.mkdir(submit_path)

    preds_test_upsampled = predict_proc(test_ids=test_ids,
                                        predict_file=predict_file)

    new_test_ids = []
    rles = []
    for n, id_ in enumerate(test_ids):
        rle = list(prob_to_rles(preds_test_upsampled[n]))
        rles.extend(rle)
        new_test_ids.extend([id_] * len(rle))

    # Create submission DataFrame
    sub = pd.DataFrame()
    sub['ImageId'] = new_test_ids
    sub['EncodedPixels'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))

    sub.to_csv(os.path.join(submit_path, r'/{}_sub.csv'.format(model_name)), index=False)


if __name__ == '__main__':
    pass
    # train_ids, test_ids = get_id()
    # predict_file = np.load(os.path.join(ROOT_DIR, r'out_files/predict/unet_32batch_100epoch_128_128/X_test.npy'))

