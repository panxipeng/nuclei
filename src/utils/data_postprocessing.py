import os
import numpy as np
from dirs import ROOT_DIR
from src.utils import data_exploration as de

def binarize(mask):
    bin_mask = mask.copy()
    bin_mask[bin_mask[:, :] <=0.8] = 0
    bin_mask[bin_mask[:, :] > 0.8] = 1

    return bin_mask

if __name__ == "__main__":

    train_ids, test_ids = de.get_id()

    preds_name = r'zf_turbo_unet_gen-1_32-bs_100-ep_0.2-vs_split-ts_128_128_X_test'
    path_to_bin_preds = os.path.join(ROOT_DIR, r'out_files/npy/bin_predict/{}'.format(preds_name))
    if not os.path.exists(path_to_bin_preds):
        os.mkdir(path_to_bin_preds)

    path_to_preds = os.path.join(ROOT_DIR, r'out_files/npy/predict/{}'.format(preds_name))

    for i, test_id in enumerate(test_ids):
        mask = np.load(os.path.join(path_to_preds, r'{}.npy'.format(test_id)))
        bin_mask = binarize(mask)
        np.save(os.path.join(path_to_bin_preds, r'{}.npy'.format(test_id)), bin_mask)