import math
import numpy as np
from enum import Enum, auto
from sklearn import linear_model, svm, tree
from sklearn.ensemble import BaggingRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

cross_folds = 10
scoring = 'neg_mean_squared_error'

class RegressionType(Enum):
    LINEAR = 'Linear Regression'
    LINEAR_BAGGING = 'Linear Regression with Bagging'
    SVM = 'SVM'
    SVM_BAGGING = 'SVM with Bagging'
    TREES = 'Regression Trees'
    TREES_BAGGING = 'Regression Trees with Bagging'
    DEEP = 'Deep Learning'

def get_model_predictions(regression_type, X, y, num_features, num_samples, skip_training=False):
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=num_features*num_samples)

    X_size = X_train.shape[0]

    reg = get_ml_model(regression_type, X_size, num_features, skip_training)

    if reg is None:
        return (None, y_test)

    reg.fit(X_train, y_train)

    predictions = reg.predict(X_test)

    return (predictions, y_test)

def get_ml_model(regression_type, X_size, num_features, skip_training=False):
    if X_size < cross_folds and not skip_training:
        return None

    if regression_type == RegressionType.LINEAR:
        reg = get_linear_regression_model()
    elif regression_type == RegressionType.LINEAR_BAGGING:
        if not skip_training:
            reg = get_trained_bagging_model(get_linear_regression_model(), X_size, num_features)
        else:
            reg = get_bagging_model(get_linear_regression_model())
    elif regression_type == RegressionType.SVM:
        if not skip_training:
            reg = get_trained_svm_model()
        else:
            reg = get_svm_model()
    elif regression_type == RegressionType.SVM_BAGGING:
        if not skip_training:
            reg = get_trained_bagging_model(get_svm_model(), X_size, num_features)
        else:
            reg = get_bagging_model(get_svm_model())
    elif regression_type == RegressionType.TREES:
        if not skip_training:
            reg = get_trained_regression_trees_model(X_size)
        else:
            reg = get_regression_trees_model()
    elif regression_type == RegressionType.TREES_BAGGING:
        if not skip_training:
            reg = get_trained_bagging_model(get_regression_trees_model(), X_size, num_features)
        else:
            reg = get_bagging_model(get_regression_trees_model())

    return reg

def get_linear_regression_model():
    return linear_model.LinearRegression()

def get_svm_model():
    return svm.SVR()

def get_trained_svm_model():
    param_grid = {
        'kernel': ['linear', 'poly', 'rbf'],
        'C': np.logspace(-2, 3, num=10),
        'gamma': np.logspace(-3, 0, num=10),
        'epsilon': np.logspace(-3, 0, num=5)
    }

    return GridSearchCV(estimator=get_svm_model(), param_grid=param_grid, cv=cross_folds, scoring=scoring)

def get_regression_trees_model():
    return tree.DecisionTreeRegressor()

def get_trained_regression_trees_model(X_size):
    param_grid = {
        'min_samples_split': np.arange(2, X_size + 1),
        'min_samples_leaf': np.arange(1, math.ceil((X_size + 1) / 3)),
        'ccp_alpha': np.logspace(-6, -2, num=10)
    }

    return GridSearchCV(estimator=get_regression_trees_model(), param_grid=param_grid, cv=cross_folds, scoring=scoring)

def get_bagging_model(base_estimator):
    return BaggingRegressor(base_estimator=base_estimator)

def get_trained_bagging_model(base_estimator, X_size, num_features):
    param_grid = {
        'n_estimators': np.arange(2, 20 + 1),
        'max_samples': np.arange(1, (X_size - math.ceil(X_size / cross_folds)) + 1),
        'max_features': np.arange(1, num_features + 1)
    }

    return GridSearchCV(estimator=get_bagging_model(base_estimator), param_grid=param_grid, cv=cross_folds, scoring=scoring)
