from error_calculations import mean_squared_error

import numpy as np
from sklearn.model_selection import train_test_split, KFold
from tensorflow.keras import models, layers, optimizers

epochs=2000
cross_folds = 10

def get_deep_model_predictions(X, y, num_features, num_samples, skip_training=False):
    # Temporary
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=num_features*num_samples)

    normalize = lambda data: np.subtract(data, np.mean(data)) / np.std(data)

    y_train = normalize(y_train)
    y_test = normalize(y_test)

    model = None

    if not skip_training:
        model = get_trained_deep_model(X_train, y_train)
    else:
        model = get_deep_model()

    model.fit(X_train, y_train, epochs=epochs, verbose=0)

    predictions = model.predict(X_test)

    return (predictions, y_test)

def get_trained_deep_model(X_train, y_train):
    mean_mse_list = []
    lr_range = np.logspace(-4, -1, num=4)
    kf = KFold(n_splits=cross_folds)

    for lr in lr_range:
        mse_list = []
        model = get_deep_model(learning_rate=lr)

        for train_indices, test_indices in kf.split(X_train, y_train):
            model.fit(X_train[train_indices], y_train[train_indices], epochs=epochs, verbose=0)

            predictions = model.predict(X_train[test_indices])

            mse = mean_squared_error(predictions, y_train[test_indices])
            mse_list.append(mse)

        mean_mse_list.append(np.mean(mse_list))

    optimal_lr = lr_range[np.argmin(mean_mse_list)]

    model = get_deep_model(learning_rate=optimal_lr)
    return model

def get_deep_model(learning_rate=0.001):
    model = models.Sequential()
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(1))

    model.compile(optimizer=optimizers.Adam(learning_rate=learning_rate), loss='mse')
    return model
