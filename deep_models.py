from error_calculations import mean_squared_error

import datetime
import os

import numpy as np
from sklearn.model_selection import KFold
from tensorflow.keras import models, layers, optimizers, callbacks

epochs = 2000
num_neurons = 128
cross_folds = 10

def fit_deep_model(X_train, y_train, skip_training=False):
    model = None

    if not skip_training:
        model = get_trained_deep_model(X_train, y_train)
    else:
        model = get_deep_model(X_train.shape[1])

    if model is None:
        return None

    callbacks = []

    # Uncomment for Tensorboard viewing
    # log_dir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    # callbacks.append(callbacks.TensorBoard(log_dir=log_dir))

    model.fit(X_train, y_train, epochs=epochs, verbose=0, callbacks=callbacks)

    return model

def get_trained_deep_model(X_train, y_train):
    if X_train.shape[0] < cross_folds:
        return None

    mean_mse_list = []
    lr_range = np.logspace(-4, -1, num=4)
    kf = KFold(n_splits=cross_folds)

    for lr in lr_range:
        mse_list = []
        model = get_deep_model(X_train.shape[1], learning_rate=lr)

        for train_indices, test_indices in kf.split(X_train, y_train):
            model.fit(X_train[train_indices], y_train[train_indices], epochs=epochs, verbose=0)

            predictions = model.predict(X_train[test_indices])

            mse = mean_squared_error(predictions, y_train[test_indices])
            mse_list.append(mse)

        mean_mse_list.append(np.mean(mse_list))

    optimal_lr = lr_range[np.argmin(mean_mse_list)]

    model = get_deep_model(X_train.shape[1], learning_rate=optimal_lr)
    return model

def get_deep_model(num_features, num_layers=10, learning_rate=0.001):
    net_layers = [
        layers.Dense(num_neurons, activation='relu', input_shape=(num_features,))
    ]

    for _ in range(0, num_layers - 1):
        net_layers.append(layers.Dense(num_neurons, activation='relu'))

    net_layers.append(layers.Dense(1))

    model = models.Sequential(net_layers)

    lr_schedule = optimizers.schedules.InverseTimeDecay(learning_rate, decay_steps=1, decay_rate=learning_rate/1000)

    model.compile(optimizer=optimizers.Adam(learning_rate=lr_schedule, clipvalue=1), loss='mse')
    return model
