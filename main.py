import numpy as np
import tensorflow as tf
from sktime.classification.distance_based import KNeighborsTimeSeriesClassifier
from sktime.datasets import load_UCR_UEA_dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sktime.datasets import load_from_arff_to_dataframe
from sktime.datatypes._panel._convert import from_nested_to_2d_array
from collections import defaultdict
from process_data import *
from shapelets_transform import*
import config
from visulization import*
from sktime.classification.deep_learning.fcn import FCNClassifier

# get data
X_train, y_train, X_test, y_test = process_ucr_dataset("Wafer")
# y_train = y_train.tolist()
# print(y_train.tolist().count('1'))
seq_length = X_train.iloc[0][0].shape[0]
fcn = FCNClassifier(n_epochs=20, batch_size=128)  
fcn.fit(X_train, y_train)  

print("Score: ", fcn.score(X_test, y_test)) # score

to_be_explained_instance_X = X_test.iloc[781][0].values.reshape(1, 1, seq_length) #781
to_be_explained_instance_y = y_test[781]
predicted_y = fcn.predict(to_be_explained_instance_X)[0]
from copy import copy
original_X = copy(to_be_explained_instance_X)
print("Predicted: ", predicted_y, "Truth: ", to_be_explained_instance_y)

target_y = 1 if int(predicted_y) == -1 else -1
# Train ShapeletTransform model
st = get_st(min_len=int(seq_length*0.1), max_len=int(seq_length*0.6))
st.fit(X_train, y_train)
st.transform(X_train)

print('>>> Target instance\'s class: ', target_y)
print()
# Index -- length: 1, start: 2, end : start+length+1
for sp in st.shapelets:
    print(sp[0], sp[1], sp[2], sp[4], sp[5])

# Get target instance's all shapelets for generator
for sp in st.shapelets:
    if (int(sp[5]) == int(predicted_y)): # Based on to-be-explained instance
        start_pos = sp[2]
        sp_length = sp[1]
        target_shapelets = get_shapelets_from_dataset(
            X_train, 
            y_train, 
            length=sp_length,
            target_class=target_y, 
            start_pos=start_pos)     
        # print(target_shapelets.shape)
        
        target_shapelets = numpy_2d_to_list(target_shapelets)
        # Generate fake shapelets
        # print("Target_shapelets: ", len(target_shapelets))
        fake_shapelets = generate_fake_shapelets_by_TimeGAN(target_shapelets, sp_length)
        # print(fake_shapelets.shape)
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
            cf_predicted = fcn.predict(to_be_explained_instance_X)[0]
            
            # print(test_X == to_be_explained_instance_X)
            
            if (int(predicted_y) != int(cf_predicted)):
                print("!!!!!!!!!!!!!!!!!!!!!!!!")
                print(start_pos, sp_length)
                print(predicted_y, cf_predicted)
                plot_time_series(to_be_explained_instance_X, original_X, start_pos, sp_length)
                # print(predicted_y, cf_predicted)
            # else:
            #     print("Not found.")
        print("End")
            # print(predicted_y, cf_predicted)
            # print()
            # print("Changed: ", int(cf_predicted) != int(predicted_label))
        # break




