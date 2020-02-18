from ml_models import get_model_predictions, RegressionType
from deep_models import get_deep_model_predictions
from error_calculations import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, symmetric_mean_absolute_percentage_error

import time
import argparse

import numpy as np
from tabulate import tabulate

def read_csv_file(file_path):
    data = np.genfromtxt(file_path, delimiter=',', skip_header=1)

    X = data[:, :-1]
    y = data[:, -1:][:, 0]
    num_features = data.shape[1] - 1

    return (X, y, num_features)

parser = argparse.ArgumentParser(description='Evaluate the prediction error for a machine learning model type and a software system.')
parser.add_argument('system', help='the software system to evaluate')
parser.add_argument('-n', help='the number of runs to repeat', type=int, default=30)
parser.add_argument('--samples', help='the set of sample sizes to run for (in multiples of n)', type=int, nargs='+', choices=[1,2,3,4,5], default=[1,2,3,4,5])
parser.add_argument('--skip-training', help='whether to skip training the machine learning models', action='store_true')
args = parser.parse_args()

max_n, samples, skip_training = args.n, args.samples, args.skip_training

file_path_to_open = 'data/' + args.system + '_AllMeasurements.csv'

(X, y, num_features) = read_csv_file(file_path_to_open)

print(args.system + ':')
print('-' * 40)

regression_types = [RegressionType.LINEAR, RegressionType.LINEAR_BAGGING, RegressionType.SVM, RegressionType.SVM_BAGGING, RegressionType.TREES, RegressionType.TREES_BAGGING, RegressionType.DEEP]

table_headings = [''] + [rt.value for rt in regression_types]

mae_table = [table_headings]
mse_table = [table_headings]
mape_table = [table_headings]
smape_table = [table_headings]
time_table = [table_headings]

for num_samples in samples:
    sample_text = str(num_samples) + 'N:'
    mae_table.append([sample_text])
    mse_table.append([sample_text])
    mape_table.append([sample_text])
    smape_table.append([sample_text])
    time_table.append([sample_text])

total_start_time = time.perf_counter()

for regression_type in regression_types:
    for sample_i, num_samples in enumerate(samples):
        errors = []

        start_time = time.perf_counter()

        for _ in range(1, max_n + 1):
            predictions, y_test = None, None

            if regression_type == RegressionType.DEEP:
                predictions, y_test = get_deep_model_predictions(X, y, num_features, num_samples, skip_training)
            else:
                predictions, y_test = get_model_predictions(regression_type, X, y, num_features, num_samples, skip_training)

            if predictions is None:
                break

            mae = mean_absolute_error(predictions, y_test)
            mse = mean_squared_error(predictions, y_test)
            mape = mean_absolute_percentage_error(predictions, y_test)
            smape = symmetric_mean_absolute_percentage_error(predictions, y_test)
            errors.append((mae, mse, mape, smape))

        # Per iteration in milliseconds
        time_elapsed = np.round(((time.perf_counter() - start_time) * 1000) / max_n, 2)

        mae_text = '-'
        mse_text = '-'
        mape_text = '-'
        smape_text = '-'
        time_text = '-'

        if len(errors) > 0:
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

            mae_text = str(mae_mean) + ' ± ' + str(mae_std)
            mse_text = str(mse_mean) + ' ± ' + str(mse_std)
            mape_text = str(mape_mean) + '% ± ' + str(mape_std) + '%'
            smape_text = str(smape_mean) + '% ± ' + str(smape_std) + '%'
            time_text = str(time_elapsed) + 'ms'

        mae_table[sample_i+1].append(mae_text)
        mse_table[sample_i+1].append(mse_text)
        mape_table[sample_i+1].append(mape_text)
        smape_table[sample_i+1].append(smape_text)
        time_table[sample_i+1].append(time_text)

        print('Completed ' + regression_type.value + ' evaluation for ' + str(num_samples) + 'N.', flush=True)

total_time_elapsed = np.round(time.perf_counter() - total_start_time, 2)

print('-' * 40)
print('Results:')
print('MAE:')
print(tabulate(mae_table, headers='firstrow', tablefmt='grid'))
print('MSE:')
print(tabulate(mse_table, headers='firstrow', tablefmt='grid'))
print('MAPE:')
print(tabulate(mape_table, headers='firstrow', tablefmt='grid'))
print('SMAPE:')
print(tabulate(smape_table, headers='firstrow', tablefmt='grid'))
print('Time elapsed:')
print(tabulate(time_table, headers='firstrow', tablefmt='grid'))
print('-' * 40)
print('Total time elapsed: ' + str(total_time_elapsed) + 's')
