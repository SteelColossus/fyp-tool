from ml_models import fit_ml_model, RegressionType
from deepperf_wrapper import fit_deep_model
from error_calculations import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, symmetric_mean_absolute_percentage_error

import time
import argparse
import pathlib
import pickle
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import psutil
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
from sklearn.model_selection import train_test_split

measuring_event = Event()
# The current process this script is running on
process = psutil.Process()


def monitor_resources():
    cpu_usages = []
    memory_usages = []

    while not measuring_event.is_set():
        with process.oneshot():
            cpu_percent = process.cpu_percent(interval=0.2)
            memory_percent = process.memory_percent()

            cpu_usages.append(cpu_percent)
            memory_usages.append(memory_percent)

    cpu_mean = np.mean(cpu_usages)
    memory_mean = np.mean(memory_usages)

    return (cpu_mean, memory_mean)


def get_system_filename(system_name):
    system_filename = None

    if system_name == 'Apache':
        system_filename = 'Apache_AllMeasurements'
    elif system_name == 'BDBC':
        system_filename = 'BDBC_AllMeasurements'
    elif system_name == 'SQL':
        system_filename = 'SQL_AllMeasurements'
    elif system_name == 'X264':
        system_filename = 'X264_AllMeasurements'
    elif system_name == 'FPGA_sort':
        system_filename = 'SS-B2'
    elif system_name == 'Apache_Storm':
        system_filename = 'SS-K1'
    elif system_name == 'LLVM':
        system_filename = 'SS-L1'
    elif system_name == 'Trimesh':
        system_filename = 'SS-M2'
    elif system_name == 'X264-DB':
        system_filename = 'SS-N1'
    elif system_name == 'SaC':
        system_filename = 'SS-O2'
    else:
        system_filename = system_name

    return system_filename


def read_csv_file(file_path):
    data = np.genfromtxt(file_path, delimiter=',', skip_header=1)

    x = data[:, :-1]
    y = data[:, -1:][:, 0]

    # Remove all features that are unused, i.e. all instances of them have the same value
    unused_features = np.all(x == x[0, :], axis=0)
    unused_indexes = np.flatnonzero(unused_features)
    num_unused_features = len(unused_indexes)

    if num_unused_features > 0:
        x = np.delete(x, unused_indexes, axis=1)
        print(f"Removed {num_unused_features} unused feature(s).")

    return (x, y)


def plot_grouped_bar_chart(filename, title, y_label, y_key, x_values, y_results, label_names, y_err_key=None):
    bar_width = 0.75 / len(label_names)
    x_intervals = np.arange(len(x_values))

    fig, ax = plt.subplots(figsize=(20, 10))
    ax.set_title(title)
    ax.set_xlabel('Regression Type')
    ax.set_ylabel(y_label)

    for index, label_name in enumerate(label_names):
        y_values = np.array([y_results[rt][index][y_key]
                             if y_results[rt][index] is not None else 0
                             for rt in regression_types])
        y_err_values = None
        capsize = None

        if y_err_key is not None:
            y_err_values = np.array([y_results[rt][index][y_err_key]
                                     if y_results[rt][index] is not None else 0
                                     for rt in regression_types])

            if not np.all(y_err_values == 0):
                capsize = 5

        # Have to set the offset of each bar here, otherwise they will stack
        ax.bar(x_intervals + (bar_width * index), y_values,
               label=label_name, width=bar_width, yerr=y_err_values, capsize=capsize)

    # Adjust the labels on the x axis to move them into the right position
    ax.set_xticks(x_intervals + (bar_width * (len(label_names) - 1) / 2))
    ax.set_xticklabels(x_values)
    ax.grid(axis='y')
    # Set the grid to appear below the bars in the chart
    ax.set_axisbelow(True)
    ax.legend()

    fig.savefig(
        f"{results_directory}/{filename}_graph.png")


parser = argparse.ArgumentParser(
    description='Evaluate the prediction error for a machine learning model type and a software system.')
parser.add_argument('system', help='the software system to evaluate')
parser.add_argument(
    '-n', help='the number of runs to repeat', type=int, default=30)
parser.add_argument('--samples', help='the set of sample sizes to run for (in multiples of n)',
                    type=int, nargs='+', choices=[1, 2, 3, 4, 5], default=[1, 2, 3, 4, 5])
parser.add_argument(
    '--skip-training', help='whether to skip training the machine learning models', action='store_true')
parser.add_argument(
    '--no-monitoring', help='whether to not monitor the CPU and memory usage', action='store_true')
args = parser.parse_args()

system_name, max_n, samples, skip_training, no_monitoring = args.system, args.n, args.samples, args.skip_training, args.no_monitoring

system_filename = get_system_filename(system_name)
file_path_to_open = f"data/{system_filename}.csv"

print(system_name + ':')
print('-' * 40)

(x, y) = read_csv_file(file_path_to_open)
num_features = x.shape[1]

regression_types = [RegressionType.LINEAR, RegressionType.LINEAR_BAGGING, RegressionType.SVM,
                    RegressionType.SVM_BAGGING, RegressionType.TREES, RegressionType.TREES_BAGGING, RegressionType.DEEP]

total_start_time = time.perf_counter()

model_results = {rt: [] for rt in regression_types}
measurement_results = {rt: [] for rt in regression_types}

for regression_type in regression_types:
    for sample_i, num_samples in enumerate(samples):
        model_results[regression_type].append([])
        measurement_results[regression_type].append({})

        cpu_percentages = []
        memory_percentages = []

        print(f"{regression_type.value}, {num_samples}N: 0/{max_n}",
              end='', flush=True)

        start_time = time.perf_counter()

        for run_i in range(1, max_n + 1):
            x_train, x_test, y_train, y_test = train_test_split(
                x, y, train_size=num_features*num_samples, random_state=run_i-1)
            max_x, max_y = None, None

            with ThreadPoolExecutor(max_workers=1) as executor:
                if not no_monitoring:
                    measuring_event.clear()
                    monitoring_thread = executor.submit(monitor_resources)

                if regression_type == RegressionType.DEEP:
                    max_x = np.amax(x_train, axis=0)

                    max_x[max_x == 0] = 1

                    max_y = np.max(y_train)

                    if max_y == 0:
                        max_y = 1

                    x_train = x_train / max_x
                    y_train = (y_train * 100) / max_y

                    model = fit_deep_model(x_train, y_train, skip_training)

                    x_train = x_train * max_x
                    y_train = (y_train * max_y) / 100
                else:
                    model = fit_ml_model(
                        regression_type, x_train, y_train, skip_training)

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

            if regression_type == RegressionType.DEEP:
                x_test = x_test / max_x

            predictions = model.predict(x_test)

            if regression_type == RegressionType.DEEP:
                x_test = x_test * max_x
                predictions = (predictions[:, 0] * max_y) / 100

            model_results[regression_type][sample_i].append({
                'actuals': y_test,
                'predictions': predictions
            })

            print('', end='\r')
            print(f"{regression_type.value}, {num_samples}N: {run_i}/{max_n}",
                  end='', flush=True)

        # Per iteration in milliseconds
        time_elapsed = np.round(
            ((time.perf_counter() - start_time) * 1000) / max_n, 2)
        measurement_results[regression_type][sample_i]['time'] = time_elapsed

        if not no_monitoring:
            cpu_percent = np.round(np.mean(cpu_percentages), 2)
            memory_percent = np.round(np.mean(memory_percentages), 2)

            measurement_results[regression_type][sample_i]['cpu'] = cpu_percent
            measurement_results[regression_type][sample_i]['memory'] = memory_percent

        print('', end='\r')
        print(
            f"Completed {regression_type.value} evaluation for {num_samples}N.", flush=True)

total_time_elapsed = np.round(time.perf_counter() - total_start_time, 2)
errors = {rt: [] for rt in regression_types}

for regression_type in regression_types:
    for sample_results in model_results[regression_type]:
        if len(sample_results) == 0:
            errors[regression_type].append(None)
            continue

        for run_results in sample_results:
            actuals = run_results['actuals']
            predictions = run_results['predictions']

            run_results['mae'] = mean_absolute_error(predictions, actuals)
            run_results['mse'] = mean_squared_error(predictions, actuals)
            run_results['mape'] = mean_absolute_percentage_error(
                predictions, actuals)
            run_results['smape'] = symmetric_mean_absolute_percentage_error(
                predictions, actuals)

        def mean(errors): return np.round(np.mean(errors), 2)
        def std(errors): return np.round(np.std(errors), 2)

        error_set = {}

        error_set['mae_mean'] = mean([result['mae']
                                      for result in sample_results])
        error_set['mse_mean'] = mean([result['mse']
                                      for result in sample_results])
        error_set['mape_mean'] = mean(
            [result['mape'] for result in sample_results])
        error_set['smape_mean'] = mean(
            [result['smape'] for result in sample_results])

        error_set['mae_std'] = std([result['mae']
                                    for result in sample_results])
        error_set['mse_std'] = std([result['mse']
                                    for result in sample_results])
        error_set['mape_std'] = std([result['mape']
                                     for result in sample_results])
        error_set['smape_std'] = std([result['smape']
                                      for result in sample_results])

        errors[regression_type].append(error_set)

table_headings = [''] + [rt.value for rt in regression_types]

tables = {'mae': [], 'mse': [], 'mape': [],
          'smape': [], 'time': [], 'cpu': [], 'memory': []}

for table in tables.values():
    table.append(table_headings)

    for sample_i, num_samples in enumerate(samples):
        sample_text = f"{num_samples}N:"
        table.append([sample_text])

for sample_i, num_samples in enumerate(samples):
    for regression_type in regression_types:
        error_set = errors[regression_type][sample_i]
        measurement_set = measurement_results[regression_type][sample_i]

        mae_text = '-'
        mse_text = '-'
        mape_text = '-'
        smape_text = '-'
        time_text = '-'
        cpu_text = '-'
        memory_text = '-'

        if error_set is not None:
            mae_text = f"{error_set['mae_mean']} +/- {error_set['mae_std']}"
            mse_text = f"{error_set['mse_mean']} +/- {error_set['mse_std']}"
            mape_text = f"{error_set['mape_mean']}% +/- {error_set['mape_std']}%"
            smape_text = f"{error_set['smape_mean']}% +/- {error_set['smape_std']}%"
            time_text = f"{measurement_set['time']}ms"

            if not no_monitoring:
                cpu_text = f"{measurement_set['cpu']}%"
                memory_text = f"{measurement_set['memory']}%"

        tables['mae'][sample_i+1].append(mae_text)
        tables['mse'][sample_i+1].append(mse_text)
        tables['mape'][sample_i+1].append(mape_text)
        tables['smape'][sample_i+1].append(smape_text)
        tables['time'][sample_i+1].append(time_text)
        tables['cpu'][sample_i+1].append(cpu_text)
        tables['memory'][sample_i+1].append(memory_text)

print('-' * 40)
print('Results:')
print('MAE:')
print(tabulate(tables['mae'], headers='firstrow', tablefmt='grid'))
print('MSE:')
print(tabulate(tables['mse'], headers='firstrow', tablefmt='grid'))
print('MAPE:')
print(tabulate(tables['mape'], headers='firstrow', tablefmt='grid'))
print('SMAPE:')
print(tabulate(tables['smape'], headers='firstrow', tablefmt='grid'))
print('Time elapsed:')
print(tabulate(tables['time'], headers='firstrow', tablefmt='grid'))

if not no_monitoring:
    print('CPU usage:')
    print(tabulate(tables['cpu'], headers='firstrow', tablefmt='grid'))
    print('Memory usage:')
    print(tabulate(tables['memory'], headers='firstrow', tablefmt='grid'))

print('-' * 40)
print(f"Total time elapsed: {total_time_elapsed}s")

formatted_date = datetime.now().strftime('%Y%m%d-%H%M%S')

results_directory = f"results/{system_name}-{formatted_date}"

print(f"Writing results to directory /{results_directory}...")
pathlib.Path(results_directory).mkdir(exist_ok=True, parents=True)

with open(f"{results_directory}/model_results.pickle", 'wb') as model_results_file:
    pickle.dump(model_results, model_results_file)

with open(f"{results_directory}/measurement_results.pickle", 'wb') as measurement_results_file:
    pickle.dump(measurement_results, measurement_results_file)

for name, table in tables.items():
    if (name == 'cpu' or name == 'memory') and no_monitoring:
        continue

    np.savetxt(f"{results_directory}/{name}_results.csv",
               table, fmt='%s', delimiter=',')

print(f"Results written to /{results_directory}.")

x_values = [rt.value for rt in regression_types]
label_names = [f"{sample}N" for sample in samples]

print("Generating graphs...")

for error, description in (('mae', 'Mean Absolute Error'), ('mse', 'Mean Squared Error'), ('mape', 'Mean Absolute Percentage Error'), ('smape', 'Symmetric Mean Absolute Percentage Error')):
    plot_grouped_bar_chart(
        error, description, error.upper(), f"{error}_mean", x_values, errors, label_names, y_err_key=f"{error}_std")

plot_grouped_bar_chart('time', 'Time Taken', 'Time per iteration (ms)',
                       'time', x_values, measurement_results, label_names)

if not no_monitoring:
    plot_grouped_bar_chart('cpu', 'CPU Usage', 'CPU (%)', 'cpu',
                           x_values, measurement_results, label_names)
    plot_grouped_bar_chart('memory', 'Memory Usage', 'Memory (%)',
                           'memory', x_values, measurement_results, label_names)

print(f"Graphs generated and saved to /{results_directory}.")
