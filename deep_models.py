from error_calculations import mean_absolute_error

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
    optimal_learning_rate_per_layer = []
    optimal_mean_mae_per_layer = []
    num_layer_range = np.arange(2, 16)
    learning_rate_range = np.logspace(-4, -1, num=4)

    for num_layer in num_layer_range:
        mae_list = []

        for learning_rate in learning_rate_range:
            model = get_deep_model(X_train.shape[1], num_layers=num_layer, learning_rate=learning_rate)

            model.fit(X_train, y_train, epochs=epochs, verbose=0)

            predictions = model.predict(X_train)
            mae = mean_absolute_error(predictions, y_train)
            mae_list.append(mae)

        optimal_learning_rate_index = np.argmin(mae_list)
        optimal_mean_mae = mae_list[optimal_learning_rate_index]
        optimal_mean_mae_per_layer.append(optimal_mean_mae)
        optimal_learning_rate = learning_rate_range[optimal_learning_rate_index]
        optimal_learning_rate_per_layer.append(optimal_learning_rate)

    optimal_num_layers_index = np.argmin(optimal_mean_mae_per_layer)
    optimal_num_layers = optimal_num_layers_index + num_layer_range[0]
    optimal_learning_rate = optimal_learning_rate_per_layer[optimal_num_layers_index]

    model = get_deep_model(X_train.shape[1], num_layers=optimal_num_layers, learning_rate=optimal_learning_rate)
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
