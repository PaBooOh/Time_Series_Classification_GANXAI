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


# Get data
dataset_name = "MoteStrain"
target_names = ['1', '-1']
X_train, y_train, X_test, y_test = process_ucr_dataset(dataset_name)
y_train = Unify_class_names(y_train, dataset_name=dataset_name)
y_test = Unify_class_names(y_test, dataset_name=dataset_name)
# print(y_train.tolist())
# show_dataset_visually(X_train, y_train, dataset_name)
positive_nums, negative_nums = get_dataset_info_group_by_class(y_train)
print("Positive nums: ", positive_nums, "Negative nums: ", negative_nums)
print()
# y_train = np.array(y_train)
seq_length = X_train.iloc[0][0].shape[0]

# Train classifier(s)
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
clf = KNeighborsTimeSeriesClassifier(distance="euclidean")

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
print("report: ")
print(classification_report(y_test, y_pred, target_names=target_names))
# print("Score: ", clf.score(X_test, y_test)) # score

# Specify a to-be-explained instance from test set.
instance_id = 71 # Test set instance
to_be_explained_instance_X = X_test.iloc[instance_id][0].values.reshape(1, 1, seq_length)
to_be_explained_instance_y = y_test[instance_id]
predicted_y = clf.predict(to_be_explained_instance_X)[0]
from copy import copy
original_X = copy(to_be_explained_instance_X)
print("To-be-explained_instance_id: ", instance_id, "Predicted: ", predicted_y, "Truth: ", to_be_explained_instance_y)
target_y = 1 if int(predicted_y) == -1 else -1

# Train RandomShapeletTransform model and get candidates
st = get_shapelet_candidates_with_ST(min_len=int(seq_length*0.1), max_len=int(seq_length*0.5))
st.fit(X_train, y_train)
st.transform(X_train)
print("To-be-explained_instance_id: ", instance_id, "Predicted: ", predicted_y, "Truth: ", to_be_explained_instance_y)
print('>>> Target instance\'s class: ', target_y, "To-be-explained_instance\'s class: ", )
print()
# Show the extracted Shapelets candidates after Shapelets Transform (ST)
# 0: information gain; 1: len; 2: start_pos; 4: instance id; 5: class; 
for sp in st.shapelets: # Index -- length: 1, start: 2, end : start+length+1
    print("Information gain: ", sp[0], "Length: ", sp[1], "Start: ", sp[2], "Instance id: ", sp[4], "Class: ", sp[5])

# Get target instance's all shapelets for generator
for sp in st.shapelets:
    if (int(sp[5]) == int(predicted_y)): # Based on to-be-explained instance
        start_pos = sp[2]
        sp_length = sp[1]
        # To get target shapelets for TimeGAN training
        target_shapelets = crop_shapelets_from_dataset(
            X_train, 
            y_train, 
            length=sp_length,
            target_class=target_y, 
            start_pos=start_pos)     
        
        target_shapelets = numpy_2d_to_list(target_shapelets) 
        # Generate fake shapelets
        fake_shapelets = generate_fake_shapelets_by_TimeGAN(target_shapelets, sp_length)
        # Concatenate, instance: (1, 1, 96)
        for fake_shapelet in fake_shapelets: # (80, 39 ,1)
            fake_shapelet = fake_shapelet.reshape(1, 1, sp_length)
            # print("Start: ", start_pos, "len: ", sp_length)
            # print("Shapelet: ", fake_shapelet)
            # print("Unchanged_interval: ", to_be_explained_instance_X[:, :, start_pos:start_pos+sp_length])
            # print("Original: ", to_be_explained_instance_X)
            # print(to_be_explained_instance_X[:, :, start_pos:start_pos+length])
            # print(fake_shapelet.shape, to_be_explained_instance_X.shape)
            to_be_explained_instance_X[:, :, start_pos:start_pos+sp_length] = fake_shapelet
            # print("Changed: ", to_be_explained_instance_X)
            cf_predicted = clf.predict(to_be_explained_instance_X)[0]
            
            if (int(predicted_y) != int(cf_predicted)):
                print("!!!!!!!!!!!!!!!!!!!!!!!!")
                print(start_pos, sp_length)
                print(predicted_y, cf_predicted)
                plot_time_series(to_be_explained_instance_X, original_X, start_pos, sp_length)
        print("End")