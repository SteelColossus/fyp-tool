import os
import csv
import time
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import svm
from sklearn import tree
from sklearn.ensemble import BaggingRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from tabulate import tabulate

class PerfData:
    def __init__(self, X, y, num_features):
        self.X = X
        self.y = y
        self.num_features = num_features

def read_csv_file(file_path):
    data = np.genfromtxt(file_path, delimiter=',', skip_header=1)

    X = data[:, :-1]
    y = data[:, -1:][:, 0]
    num_features = data.shape[1] - 1

    return PerfData(X, y, num_features)

def mean_absolute_error(predictions, actuals):
    return np.mean(np.abs(np.subtract(actuals, predictions)))

def mean_squared_error(predictions, actuals):
    return np.mean(np.square(np.subtract(actuals, predictions)))

def mean_absolute_percentage_error(predictions, actuals):
    return np.multiply(np.mean(np.abs(np.divide(np.subtract(actuals, predictions), actuals))), 100)

def symmetric_mean_absolute_percentage_error(predictions, actuals):
    return np.multiply(np.mean(np.divide(np.abs(np.subtract(actuals, predictions)), np.add(np.abs(actuals), np.abs(predictions)))), 100)

max_samples = 5
max_n = 30

folder_to_open = '../Flash-SingleConfig/Data/'

for dirpath, dirnames, filenames in os.walk(folder_to_open):
    for filename in filenames:
        if filename.startswith('sol'):
            continue

        data = read_csv_file(os.path.join(dirpath, filename))

        print(os.path.splitext(filename)[0] + ':')

        table_headings = ['', 'Linear', 'Linear with Bagging', 'SVM', 'SVM with Bagging', 'Regression Trees', 'Regression Trees with Bagging']

        mae_table = [table_headings]
        mse_table = [table_headings]
        mape_table = [table_headings]
        smape_table = [table_headings]
        time_table = [table_headings]

        for regression_type in ('linear', 'linear_bagging', 'svm', 'svm_bagging', 'trees', 'trees_bagging'):
            for num_samples in range(1, max_samples + 1):
                errors = []

                start_time = time.perf_counter()

                for i in range(1, max_n + 1):
                    X_train, X_test, y_train, y_test = train_test_split(data.X, data.y, train_size=data.num_features*num_samples)#, random_state=i-1)

                    X_size = X_train.shape[0]
                    reg = None
                    cross_folds = 3

                    if regression_type == 'linear':
                        reg = linear_model.LinearRegression()
                    elif regression_type == 'linear_bagging':
                        if X_size >= cross_folds * 2:
                            param_grid = {
                                'n_estimators': np.arange(2, 20 + 1),
                                'max_samples': np.arange(1, (X_size - math.ceil(X_size / cross_folds)) + 1),
                                'max_features': np.arange(1, data.num_features + 1)
                            }
                            reg = GridSearchCV(estimator=BaggingRegressor(base_estimator=linear_model.LinearRegression()), param_grid=param_grid, cv=cross_folds)
                        else:
                            reg = BaggingRegressor(base_estimator=linear_model.LinearRegression())
                    elif regression_type == 'svm':
                        if X_size >= cross_folds * 2:
                            param_grid = {
                                'kernel': ['linear', 'poly', 'rbf'],
                                'C': np.logspace(-2, 3, num=10),
                                'gamma': np.logspace(-3, 0, num=10),
                                'epsilon': np.logspace(-3, 0, num=5)
                            }
                            reg = GridSearchCV(estimator=svm.SVR(), param_grid=param_grid, cv=cross_folds)
                        else:
                            reg = svm.SVR()
                    elif regression_type == 'svm_bagging':
                        if X_size >= cross_folds * 2:
                            param_grid = {
                                'n_estimators': np.arange(2, 20 + 1),
                                'max_samples': np.arange(1, (X_size - math.ceil(X_size / cross_folds)) + 1),
                                'max_features': np.arange(1, data.num_features + 1)
                            }
                            reg = GridSearchCV(estimator=BaggingRegressor(base_estimator=svm.SVR()), param_grid=param_grid, cv=cross_folds)
                        else:
                            reg = BaggingRegressor(base_estimator=svm.SVR())
                    elif regression_type == 'trees':
                        if X_size >= cross_folds * 2:
                            param_grid = {
                                'min_samples_split': np.arange(2, X_size + 1),
                                'min_samples_leaf': np.arange(1, math.ceil((X_size + 1) / 3)),
                                'ccp_alpha': np.logspace(-6, -2, num=10)
                            }
                            reg = GridSearchCV(estimator=tree.DecisionTreeRegressor(), param_grid=param_grid, cv=cross_folds)
                        else:
                            reg = tree.DecisionTreeRegressor()
                    elif regression_type == 'trees_bagging':
                        if X_size >= cross_folds * 2:
                            param_grid = {
                                'n_estimators': np.arange(2, 20 + 1),
                                'max_samples': np.arange(1, (X_size - math.ceil(X_size / cross_folds)) + 1),
                                'max_features': np.arange(1, data.num_features + 1)
                            }
                            reg = GridSearchCV(estimator=BaggingRegressor(base_estimator=tree.DecisionTreeRegressor()), param_grid=param_grid, cv=cross_folds)
                        else:
                            reg = BaggingRegressor(base_estimator=tree.DecisionTreeRegressor())

                    reg.fit(X_train, y_train)

                    predictions = reg.predict(X_test)

                    mae = mean_absolute_error(predictions, y_test)
                    mse = mean_squared_error(predictions, y_test)
                    mape = mean_absolute_percentage_error(predictions, y_test)
                    smape = symmetric_mean_absolute_percentage_error(predictions, y_test)
                    errors.append((mae, mse, mape, smape))

                # Per iteration in milliseconds
                time_elapsed = np.round(((time.perf_counter() - start_time) * 1000) / max_n, 2)

                mean = lambda errors: np.round(np.mean(errors), 2)
                std = lambda errors: np.round(np.std(errors), 2)

                mae_mean = mean(np.take(errors, 0, axis=1))
                mse_mean = mean(np.take(errors, 1, axis=1))
                mape_mean = mean(np.take(errors, 2, axis=1))
                smape_mean = mean(np.take(errors, 3, axis=1))

                mae_std = std(np.take(errors, 0, axis=1))
                mse_std = std(np.take(errors, 1, axis=1))
                mape_std = std(np.take(errors, 2, axis=1))
                smape_std = std(np.take(errors, 3, axis=1))

                if len(mae_table) == num_samples:
                    mae_table.append([str(num_samples) + 'N:'])
                if len(mse_table) == num_samples:
                    mse_table.append([str(num_samples) + 'N:'])
                if len(mape_table) == num_samples:
                    mape_table.append([str(num_samples) + 'N:'])
                if len(smape_table) == num_samples:
                    smape_table.append([str(num_samples) + 'N:'])
                if len(time_table) == num_samples:
                    time_table.append([str(num_samples) + 'N:'])

                mae_table[num_samples].append(str(mae_mean) + ' ± ' + str(mae_std))
                mse_table[num_samples].append(str(mse_mean) + ' ± ' + str(mse_std))
                mape_table[num_samples].append(str(mape_mean) + '% ± ' + str(mape_std) + '%')
                smape_table[num_samples].append(str(smape_mean) + '% ± ' + str(smape_std) + '%')
                time_table[num_samples].append(str(time_elapsed) + 'ms')

        print('MAE:')
        print(tabulate(mae_table, headers='firstrow', tablefmt='fancy_grid'))
        print('MSE:')
        print(tabulate(mse_table, headers='firstrow', tablefmt='fancy_grid'))
        print('MAPE:')
        print(tabulate(mape_table, headers='firstrow', tablefmt='fancy_grid'))
        print('SMAPE:')
        print(tabulate(smape_table, headers='firstrow', tablefmt='fancy_grid'))
        print('Time elapsed:')
        print(tabulate(time_table, headers='firstrow', tablefmt='fancy_grid'))
