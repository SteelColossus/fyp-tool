from error_calculations import mean_squared_error

import numpy as np
from sklearn.model_selection import KFold
from tensorflow.keras import models, layers, optimizers

epochs=2000
cross_folds = 10

def fit_deep_model(X_train, y_train, skip_training=False):
    model = None

    if not skip_training:
        model = get_trained_deep_model(X_train, y_train)
    else:
        model = get_deep_model(X_train.shape[1])

    if model is None:
        return None

    model.fit(X_train, y_train, epochs=epochs, verbose=0)

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

def get_deep_model(num_features, learning_rate=0.001):
    model = models.Sequential([
        layers.Dense(128, activation='relu', input_shape=(num_features,)),
        layers.Dense(128, activation='relu'),
        layers.Dense(1)
    ])

    model.compile(optimizer=optimizers.Adam(learning_rate=learning_rate), loss='mse')
    return model
