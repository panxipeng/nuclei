import os
from dirs import ROOT_DIR
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split

SEED = 7

"""
                                UTIL METHODS
"""


def save_m(model, model_name):
    """
    Save trained model and weights
    :param model: Fitted model
    :param model_name: Name of the model to save
    :return: Nothing to return
    """
    model_save_path = os.path.join(ROOT_DIR, r'models')
    if not os.path.exists(model_save_path):
        os.mkdir(model_save_path)

    model_json = model.to_json()
    json_file = open(os.path.join(model_save_path, model_name + ".json"), "w")
    json_file.write(model_json)
    json_file.close()
    model.save_weights(model_save_path + '/' + model_name + ".h5")


def generator(xtr, xval, ytr, yval, batch_size):
    """
    Image generator function
    :param xtr: Training set of images in np.ndarray
    :param xval: Validation set of images in np.ndarray
    :param ytr: Training masks in np.ndarray
    :param yval: Validation masks in np.ndarray
    :param batch_size: Batch size parameter from params['batch_size']
    :return: Image generators for train and validation set of images
    """
    data_gen_args = dict(horizontal_flip=True,
                         vertical_flip=True,
                         rotation_range=90.,
                         width_shift_range=0.1,
                         height_shift_range=0.1,
                         zoom_range=0.1,
                         fill_mode='reflect')

    image_datagen = ImageDataGenerator(**data_gen_args)

    mask_datagen = ImageDataGenerator(**data_gen_args)

    image_datagen.fit(xtr, seed=SEED)
    mask_datagen.fit(ytr, seed=SEED)

    image_generator = image_datagen.flow(xtr, batch_size=batch_size, seed=SEED)
    mask_generator = mask_datagen.flow(ytr, batch_size=batch_size, seed=SEED)
    train_generator = zip(image_generator, mask_generator)

    val_gen_args = dict()
    image_datagen_val = ImageDataGenerator(**val_gen_args)
    mask_datagen_val = ImageDataGenerator(**val_gen_args)

    image_datagen_val.fit(xval, seed=SEED)
    mask_datagen_val.fit(yval, seed=SEED)

    image_generator_val = image_datagen_val.flow(xval, batch_size=batch_size, seed=SEED)
    mask_generator_val = mask_datagen_val.flow(yval, batch_size=batch_size, seed=SEED)

    val_generator = zip(image_generator_val, mask_generator_val)

    return train_generator, val_generator

def get_callbacks(filepath, verbose, patience=2):
    """
    Callbacks function: set callbacks list
    :param filepath: Path to save model checkpoint
    :param verbose: Verbose from params['verbose']
    :param patience: Amount of patient epochs before early stopping model fitting
    :return: Callbacks list
    """
    earlyStopping = EarlyStopping(monitor='acc', patience=patience, verbose=verbose)
    msave = ModelCheckpoint(filepath, verbose=verbose, save_best_only=True)

    return [earlyStopping, msave]


"""
                                FITTING MODEL
"""


def fit_save(model, model_name, X_train, Y_train, params):
    steps_per_epoch = params['steps_per_epoch']

    xtr, xval, ytr, yval = train_test_split(X_train, Y_train, test_size=0.1, random_state=7)
    train_generator, val_generator = generator(xtr, xval, ytr, yval, params['batch_size'])

    callbacks_path = os.path.join(ROOT_DIR, r'callbacks')
    if not os.path.exists(callbacks_path):
        os.mkdir(callbacks_path)

    filepath = os.path.join(callbacks_path, model_name + '_cal.h5')

    callbacks = get_callbacks(filepath=filepath, verbose=params['verbose'], patience=5)

    model.fit_generator(train_generator,
                        steps_per_epoch=steps_per_epoch,
                        epochs=params['epochs'],
                        validation_data=val_generator,
                        validation_steps=len(xval) / params['batch_size'],
                        verbose=params['verbose'],
                        callbacks=callbacks)

    print('-' * 30 + ' Saving model... ' + '-' * 30)
    save_m(model=model, model_name=model_name)
