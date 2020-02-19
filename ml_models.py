import math
import numpy as np
from enum import Enum, auto
from sklearn import linear_model, svm, tree
from sklearn.ensemble import BaggingRegressor
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

def fit_ml_model(regression_type, X_train, y_train, skip_training=False):
    model = get_ml_model(regression_type, X_train.shape[0], X_train.shape[1], skip_training)

    if model is None:
        return None

    model.fit(X_train, y_train)

    return model

def get_ml_model(regression_type, num_samples, num_features, skip_training=False):
    if num_samples < cross_folds and not skip_training:
        return None

    reg = None

    if regression_type == RegressionType.LINEAR:
        reg = get_linear_regression_model()
    elif regression_type == RegressionType.LINEAR_BAGGING:
        if not skip_training:
            reg = get_trained_bagging_model(get_linear_regression_model(), num_samples, num_features)
        else:
            reg = get_bagging_model(get_linear_regression_model())
    elif regression_type == RegressionType.SVM:
        if not skip_training:
            reg = get_trained_svm_model()
        else:
            reg = get_svm_model()
    elif regression_type == RegressionType.SVM_BAGGING:
        if not skip_training:
            reg = get_trained_bagging_model(get_svm_model(), num_samples, num_features)
        else:
            reg = get_bagging_model(get_svm_model())
    elif regression_type == RegressionType.TREES:
        if not skip_training:
            reg = get_trained_regression_trees_model(num_samples)
        else:
            reg = get_regression_trees_model()
    elif regression_type == RegressionType.TREES_BAGGING:
        if not skip_training:
            reg = get_trained_bagging_model(get_regression_trees_model(), num_samples, num_features)
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

def get_trained_regression_trees_model(num_samples):
    param_grid = {
        'min_samples_split': np.arange(2, num_samples + 1),
        'min_samples_leaf': np.arange(1, math.ceil((num_samples + 1) / 3)),
        'ccp_alpha': np.logspace(-6, -2, num=10)
    }

    return GridSearchCV(estimator=get_regression_trees_model(), param_grid=param_grid, cv=cross_folds, scoring=scoring)

def get_bagging_model(base_estimator):
    return BaggingRegressor(base_estimator=base_estimator)

def get_trained_bagging_model(base_estimator, num_samples, num_features):
    param_grid = {
        'n_estimators': np.arange(2, 20 + 1),
        'max_samples': np.arange(1, (num_samples - math.ceil(num_samples / cross_folds)) + 1),
        'max_features': np.arange(1, num_features + 1)
    }

    return GridSearchCV(estimator=get_bagging_model(base_estimator), param_grid=param_grid, cv=cross_folds, scoring=scoring)
