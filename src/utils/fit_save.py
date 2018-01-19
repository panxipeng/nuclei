import os
from keras.callbacks import EarlyStopping, ModelCheckpoint
from dirs import ROOT_DIR


"""
UTIL METHODS
"""


def save_model(model, model_name):
    model_save_path = os.path.join(ROOT_DIR, r'models')
    if not os.path.exists(model_save_path):
        os.mkdir(model_save_path)

    model_json = model.to_json()
    json_file = open(os.path.join(model_save_path, model_name + ".json"), "w")
    json_file.write(model_json)
    json_file.close()
    model.save_weights(model_save_path + '/' + model_name + ".h5")


def get_callbacks(filepath, verbose, patience=2):
   earlyStopping = EarlyStopping(patience=patience, verbose=verbose)
   msave = ModelCheckpoint(filepath, verbose=verbose, save_best_only=True)

   return [earlyStopping, msave]


"""
FITTING MODEL
"""


def fit_save(model, model_name, X_train, Y_train, params):
    callbacks_path = os.path.join(ROOT_DIR, r'callbacks')
    if not os.path.exists(callbacks_path):
        os.mkdir(callbacks_path)

    filepath = os.path.join(callbacks_path, model_name + '_cal.h5')
    # earlyStopping = EarlyStopping(patience=10, verbose=params['verbose'])
    # msave = ModelCheckpoint('model_cal.h5', verbose=params['verbose'], save_best_only=True)
    callbacks = get_callbacks(filepath=filepath, verbose=params['verbose'], patience=10)

    print('-' * 30 + ' Fitting model... ' + '-' * 30)
    model.fit(X_train, Y_train,
              validation_split=params['val_split'],
              batch_size=params['batch_size'],
              epochs=params['epochs'],
              verbose=params['verbose'],
              callbacks=callbacks)

    print('-' * 30 + ' Saving model... ' + '-' * 30)
    save_model(model=model, model_name=model_name)


if __name__ == '__main__':
    pass