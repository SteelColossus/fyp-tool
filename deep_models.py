import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras import models, layers

epochs=2000

def get_deep_model_predictions(X, y, num_features, num_samples, skip_training=False):
    # Temporary
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=num_features*num_samples)

    normalize = lambda data: np.subtract(data, np.mean(data)) / np.std(data)

    y_train = normalize(y_train)
    y_test = normalize(y_test)

    model = get_deep_model()

    model.fit(X_train, y_train, epochs=epochs)

    predictions = model.predict(X_test)

    return (predictions, y_test)

def get_deep_model():
    model = models.Sequential()
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(1))

    model.compile(optimizer='adam', loss='mse')
    return model
