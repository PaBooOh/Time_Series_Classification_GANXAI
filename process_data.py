from sktime.datasets import load_from_arff_to_dataframe
import os
import numpy as np
import pandas as pd

def process_ucr_dataset(dataset, panel=True, all_data=False):
    X_train, y_train = load_from_arff_to_dataframe('UCR_dataset/'+ dataset +'/'+ dataset +'_TRAIN.arff')
    X_test, y_test = load_from_arff_to_dataframe('UCR_dataset/'+ dataset +'/'+ dataset +'_TEST.arff')

    X = pd.concat([X_train, X_test])
    y = np.concatenate([y_train, y_test])
    if panel:
        if all_data:
            return X_train, y_train, X_test, y_test, X, y
        return X_train, y_train, X_test, y_test
    
    # return a 2d maxtrix, in which each col represents a timestamp
    from sktime.datatypes._panel._convert import from_nested_to_2d_np_array
    X_train = from_nested_to_2d_np_array(X_train)
    X_test = from_nested_to_2d_np_array(X_test)

    if all_data:
        return X_train, y_train, X_test, y_test, X, y
    return X_train, y_train, X_test, y_test

def classify_predicted_data(X_test, y_pred):
    from collections import defaultdict

    class_data = defaultdict(list)
    for i, label in enumerate(y_pred):
        class_data[label].append(X_test.iloc[i])
    class_data['1'] = np.array(class_data['1'])
    class_data['-1'] = np.array(class_data['-1'])
    return class_data

# Get the whole length of time series (i.e., the number of intervals)
def get_time_series_length(X):
    try:
        return X.iloc[0][0].shape[0]
    except:
        return X.shape[1]

# Get shapelets with specified length for a class
def get_shapelets_from_dataset(X, Y, length, start_pos, target_class):
    from sktime.datatypes._panel._convert import from_nested_to_2d_array
    X = from_nested_to_2d_array(X).to_numpy()
    res = []
    for x, y in zip(X, Y):
        if (int(y) == target_class):
            res.append(x[start_pos: start_pos+length])
    
    return np.array(res)

def numpy_2d_to_list(data):
    return [data[i].reshape(-1, 1) for i in range(data.shape[0])]

# return list
def generate_fake_shapelets_by_TimeGAN(X, seq_length):
    from ydata_synthetic.synthesizers import ModelParameters
    from ydata_synthetic.preprocessing.timeseries import processed_stock
    from ydata_synthetic.synthesizers.timeseries import TimeGAN
    gan_args = ModelParameters(batch_size=8,
                           lr=5e-4,
                           noise_dim=32,
                           layers_dim=64)
    synth = TimeGAN(model_parameters=gan_args, hidden_dim=24, seq_len=seq_length, n_seq=1, gamma=1)
    synth.train(X, train_steps=500)
    synth_data = synth.sample(len(X))
    return synth_data


# X_train, y_train, X_test, y_test = process_ucr_dataset("ECG200", panel=False)
# print(y_train.shape)