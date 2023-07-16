from sktime.datasets import load_from_arff_to_dataframe
import os
import numpy as np
import pandas as pd
import config

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

# Get shapelets with specified length and specified interval for a class
def crop_shapelets_from_dataset(X, Y, length, start_pos):
    from sktime.datatypes._panel._convert import from_nested_to_2d_array
    X = from_nested_to_2d_array(X).to_numpy()
    res = []
    for x, y in zip(X, Y):
        # if (int(y) == target_class):
        res.append(x[start_pos: start_pos + length])
    
    return res

def get_data_by_class(X, Y, label):
    from sktime.datatypes._panel._convert import from_nested_to_2d_array
    X = from_nested_to_2d_array(X).to_numpy()
    result = []
    for x, y in zip(X, Y):
        if (int(y) == int(label)):
            result.append(x)
    return result

def numpy_2d_to_list(data):
    return [data[i].reshape(-1, 1) for i in range(data.shape[0])]

# return list
## path = (e.g) /TimeGAN_Models/ECG200/ECG200_500_111.pkl
def generate_fake_sequences_by_TimeGAN(X, 
                                       seq_length, 
                                       save_model=config.to_save_model, 
                                       use_model=config.to_load_model, 
                                       path=config.model_saved_path):
    from ydata_synthetic.synthesizers import ModelParameters
    from ydata_synthetic.synthesizers.timeseries import TimeGAN
    gan_args = ModelParameters(batch_size=config.timegan_parameters["batch_size"],
                           lr=config.timegan_parameters["learning_rate"],
                           noise_dim=config.timegan_parameters["noise_dim"],
                           layers_dim=config.timegan_parameters["layers_dim"])
    
    if use_model: 
        print("Loaded pre-trained TimeGAN model successfully.")
        synth = TimeGAN.load(path) # load
    else:
        synth = TimeGAN(model_parameters=gan_args, 
                        hidden_dim=config.timegan_parameters["hidden_dim"], 
                        seq_len=seq_length, 
                        n_seq=1, 
                        gamma=config.timegan_parameters["gamma"])
        synth.train(X, train_steps=config.train_steps)
        if save_model: 
            synth.save(path) # save
            print("TimeGAN model is successully trained and saved to: ", path)
    
    # get data
    synth_data = synth.sample(len(X))
    return synth_data

def get_dataset_info_group_by_class(y_train):
    dataset_size = len(y_train)
    nums_positive = y_train.tolist().count('1')
    return nums_positive, dataset_size-nums_positive

# Used for transfer label 0 to -1.
def Unify_class_names(y_train, dataset_name):
    y_train = y_train.tolist() 
    if dataset_name == "BinaryHeartbeat":
        y_train = ['1' if x == 'Normal' else '-1' for x in y_train]
    elif dataset_name == "Earthquakes":
        y_train = ['-1' if x == '0' else x for x in y_train]
    elif dataset_name == "HouseTwenty":
        y_train = ['-1' if x == '2' else x for x in y_train]
    elif dataset_name == "SharePriceIncrease": 
        y_train = ['-1' if x == '0' else x for x in y_train]
    elif dataset_name == "MoteStrain": 
        y_train = ['-1' if x == '2' else x for x in y_train]
    return np.array(y_train)
# X_train, y_train, X_test, y_test = process_ucr_dataset("ECG200", panel=False)
# print(y_train.shape)