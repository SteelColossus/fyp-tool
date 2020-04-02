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


def fit_ml_model(regression_type, x_train, y_train, skip_training=False):
    model = get_ml_model(
        regression_type, x_train.shape[0], x_train.shape[1], skip_training)

    if model is None:
        return None

    model.fit(x_train, y_train)

    return model


def get_ml_model(regression_type, num_samples, num_features, skip_training=False):
    if num_samples < cross_folds and not skip_training and regression_type != RegressionType.LINEAR:
        return None

    reg = None

    if regression_type == RegressionType.LINEAR:
        reg = get_linear_regression_model()
    elif regression_type == RegressionType.LINEAR_BAGGING:
        if not skip_training:
            reg = get_trained_bagging_model(
                get_linear_regression_model(), num_samples, num_features)
        else:
            reg = get_bagging_model(get_linear_regression_model())
    elif regression_type == RegressionType.SVM:
        if not skip_training:
            reg = get_trained_svm_model()
        else:
            reg = get_svm_model()
    elif regression_type == RegressionType.SVM_BAGGING:
        if not skip_training:
            reg = get_trained_bagging_model(
                get_svm_model(), num_samples, num_features)
        else:
            reg = get_bagging_model(get_svm_model())
    elif regression_type == RegressionType.TREES:
        if not skip_training:
            reg = get_trained_regression_trees_model(num_samples)
        else:
            reg = get_regression_trees_model()
    elif regression_type == RegressionType.TREES_BAGGING:
        if not skip_training:
            reg = get_trained_bagging_model(
                get_regression_trees_model(), num_samples, num_features)
        else:
            reg = get_bagging_model(get_regression_trees_model())

    return reg


def get_linear_regression_model():
    return linear_model.LinearRegression()


def get_svm_model():
    # These are the default hyperparameters for version 0.22 of scikit-learn: https://scikit-learn.org/0.22/modules/generated/sklearn.svm.SVR.html
    # They are set here for clarity and compatibility with other versions
    return svm.SVR(kernel='rbf', C=1.0, gamma='scale', epsilon=0.1)


def get_trained_svm_model():
    # The linear kernel can take a long amount of time when the regularization parameter C is high, so they are separated out here
    param_grid = [
        {
            'kernel': ['linear'],
            'C': np.logspace(-2, 1, num=10),
            'gamma': np.logspace(-3, 0, num=10),
            'epsilon': np.logspace(-3, 0, num=5)
        },
        {
            'kernel': ['poly', 'rbf'],
            'C': np.logspace(-2, 3, num=10),
            'gamma': np.logspace(-3, 0, num=10),
            'epsilon': np.logspace(-3, 0, num=5)
        }
    ]

    return GridSearchCV(estimator=get_svm_model(), param_grid=param_grid, cv=cross_folds, scoring=scoring)


def get_regression_trees_model():
    # These are the default hyperparameters for version 0.22 of scikit-learn: https://scikit-learn.org/0.22/modules/generated/sklearn.tree.DecisionTreeRegressor.html
    # They are set here for clarity and compatibility with other versions
    return tree.DecisionTreeRegressor(min_samples_split=2, min_samples_leaf=1, ccp_alpha=0.0)


def get_trained_regression_trees_model(num_samples):
    param_grid = []

    base_param_grid = {
        'ccp_alpha': np.logspace(-6, -2, num=10)
    }

    for min_samples_split in range(2, num_samples + 1):
        min_samples_leaf = int(min_samples_split / 3)

        if min_samples_leaf <= 0:
            min_samples_leaf = 1

        new_param_grid = base_param_grid.copy()
        new_param_grid['min_samples_split'] = [min_samples_split]
        new_param_grid['min_samples_leaf'] = [min_samples_leaf]

        param_grid.append(new_param_grid)

    return GridSearchCV(estimator=get_regression_trees_model(), param_grid=param_grid, cv=cross_folds, scoring=scoring)


def get_bagging_model(base_estimator):
    return BaggingRegressor(base_estimator=base_estimator)


def get_trained_bagging_model(base_estimator, num_samples, num_features):
    param_grid = {
        'n_estimators': np.arange(2, 20 + 1)
    }

    return GridSearchCV(estimator=get_bagging_model(base_estimator), param_grid=param_grid, cv=cross_folds, scoring=scoring)
