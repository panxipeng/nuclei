from src.nn_var.mask_rcnn.config import Config
import src.nn_var.mask_rcnn.utils as utils
import os
import numpy as np
import skimage.io


class CellsConfig(Config):

    # Give the configuration a recognizable name
    NAME = "cell"
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    STEPS_PER_EPOCH = 1200 // (IMAGES_PER_GPU * GPU_COUNT)
    VALIDATION_STEPS = 120 // (IMAGES_PER_GPU * GPU_COUNT)
    LEARNING_RATE = 0.004
    TRAIN_ROIS_PER_IMAGE = 512
    NUM_CLASSES = 1 + 1  # background + cells class
    DETECTION_MAX_INSTANCES = 400
    # RPN_ANCHOR_SCALES = (16, 32, 64, 128, 256)
    DETECTION_NMS_THRESHOLD = 0.1
    DETECTION_MIN_CONFIDENCE = 0.1
    MAX_GT_INSTANCES = 250
    RPN_NMS_THRESHOLD = 0.95


class CellsDataset(utils.Dataset):

    def load_cells(self, dataset_dir, image_ids):
        self.add_class("cells", 1, "cell")
        for i in image_ids:
            self.add_image("cells", image_id=i,
                           # width=width, height=height,
                           path=os.path.join(dataset_dir, r"{}/images/{}.png".format(i, i)),
                           path_mask=os.path.join(dataset_dir, i, 'masks'))

    def load_mask(self, image_id):

        instance_masks = []
        path_mask = self.image_info[image_id]['path_mask']
        masks_names = next(os.walk(path_mask))[2]

        for i, mask in enumerate(masks_names):
            if mask.split('.')[-1] != 'png':
                continue
            img = skimage.io.imread(os.path.join(path_mask, mask))
            instance_masks.append(img)

        masks = np.stack(instance_masks, axis=2)
        class_ids = np.ones(shape=(len(masks_names)), dtype=np.int32)

        return masks, class_ids

