import experiment
import experiment
import numpy as np
import tensorflow as tf
import config

from process_data import *
from shapelets_transform import*
from sklearn.metrics import classification_report
from visulization import *

"""
Configuration
"""
dataset_name = config.dataset_name
class_names = config.class_names
classifier_name = config.classifier_name
random_seed = config.random_seed

"""
1 - Process dataset and get processed data
"""
X_train, y_train, X_test, y_test = process_ucr_dataset(dataset_name)
y_train = Unify_class_names(y_train, dataset_name=dataset_name)
y_test = Unify_class_names(y_test, dataset_name=dataset_name)
# show_dataset_visually(X_train, y_train, dataset_name)
positive_nums, negative_nums = get_dataset_info_group_by_class(y_train)
print("Positive nums: ", positive_nums, "Negative nums: ", negative_nums)
print()
seq_length = X_train.iloc[0][0].shape[0]


"""
2 - Train classifier(s)
"""
from sktime.classification.interval_based import DrCIF
from sktime.classification.feature_based import Catch22Classifier
from sklearn.ensemble import RandomForestClassifier
from sktime.classification.distance_based import KNeighborsTimeSeriesClassifier
from sktime.classification.deep_learning.cnn import CNNClassifier
assert classifier_name in ["KNN", "CNN", "DrCIF", "Catch22"]
clf = None
if classifier_name == "KNN":
    # (1) KNN-l2
    clf = KNeighborsTimeSeriesClassifier(distance="euclidean")
elif classifier_name == "CNN":
    # (2) CNN
    clf = CNNClassifier(random_state=random_seed)
elif classifier_name == "DrCIF":
    # (3) DrCIF
    clf = DrCIF(
        n_estimators=3, n_intervals=2, att_subsample_size=2, random_state=random_seed
    )
elif classifier_name == "Catch22":
    # (4) Catch22
    clf = Catch22Classifier(
        estimator=RandomForestClassifier(n_estimators=5),
        outlier_norm=True,
        random_state=random_seed
    ) 
 
print("(1) Loading classifier ......")
print()

# from zipfile import ZipFile
# with ZipFile(config.classifier_path, 'r') as myzip:
#     serialized_object = myzip.open("keras/saved_model.pb")

# clf.load_from_serial(serialized_object)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Report: ")
print(classification_report(y_test, y_pred, target_names=class_names))
# print("Score: ", clf.score(X_test, y_test)) # score
# clf.save(config.classifier_path)
print()
"""
3 - Comparative Study: Alibi cf
"""
# Isolation forest
from sktime.datatypes._panel._convert import from_nested_to_2d_array
X_train_float = from_nested_to_2d_array(X_train).to_numpy()
if_model = experiment.train_isolation_forest(X_train_float, random_seed)
print("Starting carrying out experiment on Alibi")
instance_id = config.instance_id # Test set instance
to_be_explained_instance = X_test.iloc[instance_id][0].values.reshape(1, seq_length)
to_be_explained_instance_label_y = y_test[instance_id]
to_be_explained_instance_predicted_y = clf.predict(to_be_explained_instance)[0]
print("To-be-explained_instance_id: ", instance_id, ", Predicted: ", to_be_explained_instance_predicted_y, ", Truth: ", to_be_explained_instance_label_y)
probs = clf.predict_proba(to_be_explained_instance)
print(to_be_explained_instance.shape)
# print(probs)
# pred_class = probs.argmax(axis=1).item()
# pred_prob = probs.max(axis=1).item()
# print(pred_prob, pred_class)
alibi = experiment.AlibiExperiment(classifier=clf, shape=to_be_explained_instance.shape, pred_proba=probs)
explanation = alibi.get_explanation(to_be_explained_instance, probs)
print(explanation)
closeness_l1 = experiment.calculate_closeness(explanation.cf['X'], to_be_explained_instance, "l1")
closeness_l2 = experiment.calculate_closeness(explanation.cf['X'], to_be_explained_instance, "l2")
sparsity = experiment.calculate_sparsity(explanation.cf['X'], to_be_explained_instance)
isolation_predict = experiment.predict_outlier_with_isolation_forest(if_model, explanation.cf['X'])
print("Closeness_L1: ", closeness_l1, ", Closeness_L2: ", closeness_l2, ", Sparsity: ", sparsity, ", Isolation: ", isolation_predict)





