from ml_models import fit_ml_model, RegressionType
from deep_models import fit_deep_model
from error_calculations import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, symmetric_mean_absolute_percentage_error

import time
import argparse
import psutil
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import numpy as np
from tabulate import tabulate
from sklearn.model_selection import train_test_split

measuring_event = Event()

def monitor_resources():
    cpu_usages = []
    memory_usages = []

    while not measuring_event.is_set():
        cpu_percent = psutil.cpu_percent(interval=0.2)
        memory_percent = psutil.virtual_memory().percent

        cpu_usages.append(cpu_percent)
        memory_usages.append(memory_percent)

    cpu_mean = np.mean(cpu_usages)
    memory_mean = np.mean(memory_usages)

    return (cpu_mean, memory_mean)

def read_csv_file(file_path):
    data = np.genfromtxt(file_path, delimiter=',', skip_header=1)

    X = data[:, :-1]
    y = data[:, -1:][:, 0]

    return (X, y)

parser = argparse.ArgumentParser(description='Evaluate the prediction error for a machine learning model type and a software system.')
parser.add_argument('system', help='the software system to evaluate')
parser.add_argument('-n', help='the number of runs to repeat', type=int, default=30)
parser.add_argument('--samples', help='the set of sample sizes to run for (in multiples of n)', type=int, nargs='+', choices=[1,2,3,4,5], default=[1,2,3,4,5])
parser.add_argument('--skip-training', help='whether to skip training the machine learning models', action='store_true')
parser.add_argument('--no-monitoring', help='whether to not monitor the CPU and memory usage', action='store_true')
args = parser.parse_args()

max_n, samples, skip_training, no_monitoring = args.n, args.samples, args.skip_training, args.no_monitoring

file_path_to_open = 'data/' + args.system + '_AllMeasurements.csv'

(X, y) = read_csv_file(file_path_to_open)
num_features = X.shape[1]

print(args.system + ':')
print('-' * 40)

regression_types = [RegressionType.LINEAR, RegressionType.LINEAR_BAGGING, RegressionType.SVM, RegressionType.SVM_BAGGING, RegressionType.TREES, RegressionType.TREES_BAGGING, RegressionType.DEEP]

total_start_time = time.perf_counter()

model_results = {rt: [] for rt in regression_types}
measurement_results = {rt: [] for rt in regression_types}

for regression_type in regression_types:
    for sample_i, num_samples in enumerate(samples):
        model_results[regression_type].append([])
        measurement_results[regression_type].append({})

        cpu_percentages = []
        memory_percentages = []

        start_time = time.perf_counter()

        for run_i in range(1, max_n + 1):
            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=num_features*num_samples, random_state=run_i-1)

            with ThreadPoolExecutor(max_workers=1) as executor:
                if not no_monitoring:
                    measuring_event.clear()
                    monitoring_thread = executor.submit(monitor_resources)

                if regression_type == RegressionType.DEEP:
                    max_y = np.max(y_train)

                    y_train = (y_train / max_y) * 100
                    y_test = (y_test / max_y) * 100

                    model = fit_deep_model(X_train, y_train, skip_training)

                    y_train = (y_train / 100) * max_y
                    y_test = (y_test / 100) * max_y
                else:
                    model = fit_ml_model(regression_type, X_train, y_train, skip_training)

                if not no_monitoring:
                    measuring_event.set()

                    try:
                        cpu_mean, memory_mean = monitoring_thread.result()
                        cpu_percentages.append(cpu_mean)
                        memory_percentages.append(memory_mean)
                    except Exception as ex:
                        print(ex)

            if model is None:
                break

            predictions = model.predict(X_test)

            model_results[regression_type][sample_i].append({
                'actuals': y_test,
                'predictions': predictions
            })

            print('.', end='', flush=True)

        # Per iteration in milliseconds
        time_elapsed = np.round(((time.perf_counter() - start_time) * 1000) / max_n, 2)
        measurement_results[regression_type][sample_i]['time'] = time_elapsed

        if not no_monitoring:
            cpu_percent = np.round(np.mean(cpu_percentages), 2)
            memory_percent = np.round(np.mean(memory_percentages), 2)

            measurement_results[regression_type][sample_i]['cpu'] = cpu_percent
            measurement_results[regression_type][sample_i]['memory'] = memory_percent

        print('', end='\r')
        print('Completed ' + regression_type.value + ' evaluation for ' + str(num_samples) + 'N.', flush=True)

total_time_elapsed = np.round(time.perf_counter() - total_start_time, 2)
errors = {rt: [] for rt in regression_types}

for regression_type in regression_types:
    for sample_results in model_results[regression_type]:
        if len(sample_results) == 0:
            errors[regression_type].append(None)
            break

        for run_results in sample_results:
            actuals = run_results['actuals']
            predictions = run_results['predictions']

            run_results['mae'] = mean_absolute_error(predictions, actuals)
            run_results['mse'] = mean_squared_error(predictions, actuals)
            run_results['mape'] = mean_absolute_percentage_error(predictions, actuals)
            run_results['smape'] = symmetric_mean_absolute_percentage_error(predictions, actuals)

        mean = lambda errors: np.round(np.mean(errors), 2)
        std = lambda errors: np.round(np.std(errors), 2)

        error_set = {}

        error_set['mae_mean'] = mean([result['mae'] for result in sample_results])
        error_set['mse_mean'] = mean([result['mse'] for result in sample_results])
        error_set['mape_mean'] = mean([result['mape'] for result in sample_results])
        error_set['smape_mean'] = mean([result['smape'] for result in sample_results])

        error_set['mae_std'] = std([result['mae'] for result in sample_results])
        error_set['mse_std'] = std([result['mse'] for result in sample_results])
        error_set['mape_std'] = std([result['mape'] for result in sample_results])
        error_set['smape_std'] = std([result['smape'] for result in sample_results])

        errors[regression_type].append(error_set)

table_headings = [''] + [rt.value for rt in regression_types]

mae_table = [table_headings]
mse_table = [table_headings]
mape_table = [table_headings]
smape_table = [table_headings]
time_table = [table_headings]

if not no_monitoring:
    cpu_table = [table_headings]
    memory_table = [table_headings]

for sample_i, num_samples in enumerate(samples):
    sample_text = f"{num_samples}N:"
    mae_table.append([sample_text])
    mse_table.append([sample_text])
    mape_table.append([sample_text])
    smape_table.append([sample_text])
    time_table.append([sample_text])

    if not no_monitoring:
        cpu_table.append([sample_text])
        memory_table.append([sample_text])

    for regression_type in regression_types:
        error_set = errors[regression_type][sample_i]
        measurement_set = measurement_results[regression_type][sample_i]

        mae_text = '-'
        mse_text = '-'
        mape_text = '-'
        smape_text = '-'
        time_text = '-'

        if not no_monitoring:
            cpu_text = '-'
            memory_text = '-'

        if error_set is not None:
            mae_text = f"{error_set['mae_mean']} ± {error_set['mae_std']}"
            mse_text = f"{error_set['mse_mean']} ± {error_set['mse_std']}"
            mape_text = f"{error_set['mape_mean']}% ± {error_set['mape_std']}%"
            smape_text = f"{error_set['smape_mean']}% ± {error_set['smape_std']}%"
            time_text = f"{measurement_set['time']}ms"
            
            if not no_monitoring:
                cpu_text = f"{measurement_set['cpu']}%"
                memory_text = f"{measurement_set['memory']}%"

        mae_table[sample_i+1].append(mae_text)
        mse_table[sample_i+1].append(mse_text)
        mape_table[sample_i+1].append(mape_text)
        smape_table[sample_i+1].append(smape_text)
        time_table[sample_i+1].append(time_text)

        if not no_monitoring:
            cpu_table[sample_i+1].append(cpu_text)
            memory_table[sample_i+1].append(memory_text)

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

if not no_monitoring:
    print('CPU usage:')
    print(tabulate(cpu_table, headers='firstrow', tablefmt='grid'))
    print('Memory usage:')
    print(tabulate(memory_table, headers='firstrow', tablefmt='grid'))

print('-' * 40)
print(f"Total time elapsed: {total_time_elapsed}s")
