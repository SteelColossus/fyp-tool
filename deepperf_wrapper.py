import sys
from os import path
# In order to properly import DeepPerf, due to it not using relative paths, we have to add the directory to sys.path so it can find the imports
sys.path.append(path.join(path.dirname(path.abspath(__file__)), 'extensions', 'DeepPerf'))
from extensions.DeepPerf.AutoDeepPerf import nn_l1_val, MLPPlainModel, MLPSparseModel

import numpy as np

def fit_deep_model(x_train, y_train, skip_training=False):
    y_train = y_train[:, np.newaxis]
    N_train, n = x_train.shape

    # Set some defaults if training is skipped
    n_layer_opt = 10
    lambda_f = 0.01
    lr_opt = 0.001

    if not skip_training:
    ### THE FOLLOWING CODE IS TAKEN FROM THE DEEPPERF TOOL ###
        # Split train data into 2 parts (67-33)
        N_cross = int(np.ceil(N_train*2/3))
        X_train1 = x_train[0:N_cross, :]
        Y_train1 = y_train[0:N_cross]
        X_train2 = x_train[N_cross:N_train, :]
        Y_train2 = y_train[N_cross:N_train]

        # Choosing the right number of hidden layers and , start with 2
        # The best layer is when adding more layer and the testing error
        # does not increase anymore
        config = dict()
        config['num_input'] = n
        config['num_neuron'] = 128
        config['lambda'] = 'NA'
        config['decay'] = 'NA'
        config['verbose'] = 0
        dir_output = 'C:/Users/Downloads'
        abs_error_all = np.zeros((15, 4))
        abs_error_all_train = np.zeros((15, 4))
        abs_error_layer_lr = np.zeros((15, 2))
        abs_err_layer_lr_min = 100
        count = 0
        layer_range = range(2, 15)
        lr_range = np.logspace(np.log10(0.0001), np.log10(0.1), 4)
        for n_layer in layer_range:
            config['num_layer'] = n_layer
            for lr_index, lr_initial in enumerate(lr_range):
                model = MLPPlainModel(config, dir_output)
                model.build_train()
                model.train(X_train1, Y_train1, lr_initial)

                Y_pred_train = model.predict(X_train1)
                abs_error_train = np.mean(np.abs(Y_pred_train - Y_train1))
                abs_error_all_train[int(n_layer), lr_index] = abs_error_train

                Y_pred_val = model.predict(X_train2)
                abs_error = np.mean(np.abs(Y_pred_val - Y_train2))
                abs_error_all[int(n_layer), lr_index] = abs_error

            # Pick the learning rate that has the smallest train cost
            # Save testing abs_error correspond to the chosen learning_rate
            temp = abs_error_all_train[int(n_layer), :]/np.max(abs_error_all_train)
            temp_idx = np.where(abs(temp) < 0.0001)[0]
            if len(temp_idx) > 0:
                lr_best = lr_range[np.max(temp_idx)]
                err_val_best = abs_error_all[int(n_layer), np.max(temp_idx)]
            else:
                lr_best = lr_range[np.argmin(temp)]
                err_val_best = abs_error_all[int(n_layer), np.argmin(temp)]

            abs_error_layer_lr[int(n_layer), 0] = err_val_best
            abs_error_layer_lr[int(n_layer), 1] = lr_best

            if abs_err_layer_lr_min >= abs_error_all[int(n_layer), np.argmin(temp)]:
                abs_err_layer_lr_min = abs_error_all[int(n_layer),
                                                        np.argmin(temp)]
                count = 0
            else:
                count += 1

            if count >= 2:
                break
        abs_error_layer_lr = abs_error_layer_lr[abs_error_layer_lr[:, 1] != 0]

        # Get the optimal number of layers
        n_layer_opt = layer_range[np.argmin(abs_error_layer_lr[:, 0])]+5

        # Find the optimal learning rate of the specific layer
        config['num_layer'] = n_layer_opt
        for lr_index, lr_initial in enumerate(lr_range):
            model = MLPPlainModel(config, dir_output)
            model.build_train()
            model.train(X_train1, Y_train1, lr_initial)

            Y_pred_train = model.predict(X_train1)
            abs_error_train = np.mean(np.abs(Y_pred_train - Y_train1))
            abs_error_all_train[int(n_layer), lr_index] = abs_error_train

            Y_pred_val = model.predict(X_train2)
            abs_error = np.mean(np.abs(Y_pred_val - Y_train2))
            abs_error_all[int(n_layer), lr_index] = abs_error

        temp = abs_error_all_train[int(n_layer), :]/np.max(abs_error_all_train)
        temp_idx = np.where(abs(temp) < 0.0001)[0]
        if len(temp_idx) > 0:
            lr_best = lr_range[np.max(temp_idx)]
        else:
            lr_best = lr_range[np.argmin(temp)]

        lr_opt = lr_best

        # Use grid search to find the right value of lambda
        lambda_range = np.logspace(-2, np.log10(1000), 30)
        error_min = np.zeros((1, len(lambda_range)))
        rel_error_min = np.zeros((1, len(lambda_range)))
        for idx, lambd in enumerate(lambda_range):
            val_abserror, val_relerror = nn_l1_val(X_train1, Y_train1,
                                                    X_train2, Y_train2,
                                                    n_layer_opt, lambd, lr_opt)
            error_min[0, idx] = val_abserror
            rel_error_min[0, idx] = val_relerror

        # Find the value of lambda that minimize error_min
        lambda_f = lambda_range[np.argmin(error_min)]

    # Solve the final NN with the chosen lambda_f on the training data
    config = dict()
    config['num_neuron'] = 128
    config['num_input'] = n
    config['num_layer'] = n_layer_opt
    config['lambda'] = lambda_f
    config['verbose'] = 0
    dir_output = 'C:/Users/Downloads'
    model = MLPSparseModel(config, dir_output)
    model.build_train()
    model.train(x_train, y_train, lr_opt)
    ##########################################################

    return model
