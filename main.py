# from sktime.datasets import load_UCR_UEA_dataset
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# from sktime.datasets import load_from_arff_to_dataframe
# from sktime.datatypes._panel._convert import from_nested_to_2d_array
import numpy as np
import tensorflow as tf
from sktime.classification.distance_based import KNeighborsTimeSeriesClassifier
from collections import defaultdict
from process_data import *
from shapelets_transform import*
from sklearn.metrics import classification_report
import config
from visulization import*



"""
Get data
"""
dataset_name = "ECG200"
target_names = ['1', '-1']
X_train, y_train, X_test, y_test = process_ucr_dataset(dataset_name)
y_train = Unify_class_names(y_train, dataset_name=dataset_name)
y_test = Unify_class_names(y_test, dataset_name=dataset_name)
# print(y_train.tolist())
# show_dataset_visually(X_train, y_train, dataset_name)
positive_nums, negative_nums = get_dataset_info_group_by_class(y_train)
print("Positive nums: ", positive_nums, "Negative nums: ", negative_nums)
# print(len(get_data_by_class(X_train, y_train, '-1')))
print()
# y_train = np.array(y_train)
seq_length = X_train.iloc[0][0].shape[0]

"""
Train classifier(s)
"""
from sktime.classification.interval_based import DrCIF
from sktime.classification.feature_based import Catch22Classifier
from sklearn.ensemble import RandomForestClassifier
from sktime.classification.distance_based import KNeighborsTimeSeriesClassifier
from sktime.classification.deep_learning.cnn import CNNClassifier
# from sktime.classification.kernel_based import RocketClassifier
# from sktime.classification.deep_learning.lstmfcn import LSTMFCNClassifier
# from sktime.classification.dictionary_based import BOSSEnsemble
# from sktime.classification.interval_based import CanonicalIntervalForest
# fcn = CanonicalIntervalForest(
#     n_estimators=3, n_intervals=2, att_subsample_size=2
# )
# fcn = FCNClassifier(n_epochs=20, batch_size=128)  
# clf = BOSSEnsemble(max_ensemble_size=3, random_state=111) 

# (1) KNN-l2
clf = KNeighborsTimeSeriesClassifier()

# (2) CNN
# fcn = CNNClassifier(random_state=111)

# (3) DrCIF
# clf = DrCIF(
#     n_estimators=3, n_intervals=2, att_subsample_size=2
# )

# (4) Catch22
# clf = Catch22Classifier(
#     estimator=RandomForestClassifier(n_estimators=5),
#     outlier_norm=True,
#     random_state=111
# ) 
 
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Report: ")
print(classification_report(y_test, y_pred, target_names=target_names))
# print("Score: ", clf.score(X_test, y_test)) # score

# Specify a to-be-explained instance from test set.
instance_id = 70 # Test set instance
to_be_explained_instance = X_test.iloc[instance_id][0].values.reshape(1, 1, seq_length)
to_be_explained_instance_y = y_test[instance_id]
to_be_explained_instance_predicted_y = clf.predict(to_be_explained_instance)[0]
from copy import copy
# original_instance = copy(to_be_explained_instance)
print("To-be-explained_instance_id: ", instance_id, ", Predicted: ", to_be_explained_instance_predicted_y, ", Truth: ", to_be_explained_instance_y)
target_y = 1 if int(to_be_explained_instance_predicted_y) == -1 else -1

"""
Train RandomShapeletTransform model and get candidates
"""
st = get_shapelet_candidates_with_ST(min_len=int(seq_length*0.1), max_len=int(seq_length*0.5))
st.fit(X_train, y_train)
st.transform(X_train)
print('>>> Target instance\'s class: ', target_y, "To-be-explained_instance\'s class: ", )
print()
# Show the extracted Shapelets candidates after Shapelets Transform (ST)
# 0: information gain; 1: len; 2: start_pos; 4: instance id; 5: class; 
sorted_shapelets = sorted(st.shapelets, key=lambda sp: sp[1])
for sp_idx, sp in enumerate(sorted_shapelets): # Index -- length: 1, start: 2, end : start+length+1
    print("Information gain: ", sp[0], ", Start: ", sp[2], ", Length: ", sp[1], ", From Instance id: ", sp[4], ", Class: ", sp[5])

"""
Get target instance's all shapelets for generator
"""
target_instances = get_data_by_class(X_train, y_train, target_y)
fake_target_instances = generate_fake_sequences_by_TimeGAN(target_instances, seq_length)
print(fake_target_instances.shape)
# for sp in sorted_shapelets:
# for fake_instance in fake_target_instances:
for sp in sorted_shapelets:
    if (int(sp[5]) == int(to_be_explained_instance_predicted_y)): # Look for the discriminative area of to-be-explained instance
        start_pos = sp[2]
        sp_length = sp[1]
        # Crop a specified intervals from target instances to get target shapelets for TimeGAN training
        fake_shapelets = crop_shapelets_from_dataset(
            X_train, 
            y_train, 
            length=sp_length,
            start_pos=start_pos)     
        print("TimeGAN_Shape: ", fake_shapelets.shape)
        fake_shapelets = numpy_2d_to_list(fake_shapelets) 
        # Generate fake shapelets with the help of TIMEGAN
        # fake_shapelets = generate_fake_sequences_by_TimeGAN(target_shapelets, sp_length)
        # Concatenate, instance: (1, 1, 96)
        
        for time_gan_idx, fake_shapelet in enumerate(fake_shapelets): # (80, 39 ,1)
            fake_shapelet = fake_shapelet.reshape(1, 1, sp_length)
            cf_instance = copy(to_be_explained_instance)
            # print("Start: ", start_pos, "len: ", sp_length)
            # print("Shapelet: ", fake_shapelet)
            # print("Unchanged_interval: ", to_be_explained_instance_X[:, :, start_pos:start_pos+sp_length])
            # print("Original: ", to_be_explained_instance_X)
            # print(to_be_explained_instance_X[:, :, start_pos:start_pos+length])
            # print(fake_shapelet.shape, to_be_explained_instance_X.shape)
            cf_instance[:, :, start_pos:start_pos+sp_length] = fake_shapelet
            # print("Changed: ", to_be_explained_instance_X)
            cf_predicted = clf.predict(cf_instance)[0]
            
            if (int(to_be_explained_instance_predicted_y) != int(cf_predicted)):
                print("TimeGAN_id: ", time_gan_idx)
                print("Id: ", sp_id, "Information gain: ", sp[0], ", Start: ", sp[2], ", Length: ", sp[1], ", From Instance id: ", sp[4], ", Class: ", sp[5])
                print()
                plot_time_series(cf_instance, to_be_explained_instance, start_pos, sp_length)
        print("End")
        print()