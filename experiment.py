import matplotlib.pyplot as plt
import numpy as np
import json
import os
# from alibi.explainers import Counterfactual
from scipy.interpolate import UnivariateSpline
from config import experiment_result_path_aggregate, experiment_result_path_individual, alibi_parameters


def shapelet_example():
    time_series = np.random.rand(100)
    time_steps = np.arange(1, 101)
    # UnivariateSpline smooth
    spl = UnivariateSpline(time_steps, time_series, s=1.0)
    time_steps_new = np.linspace(1, 100, 500)
    plt.plot(time_steps_new[0:250], spl(time_steps_new[0:250]), color='blue')
    plt.plot(time_steps_new[249:375], spl(time_steps_new[249:375]), color='green', alpha=0.5)
    plt.plot(time_steps_new[374:], spl(time_steps_new[374:]), color='blue')
    plt.show()


def calculate_closeness(instance_A, instance_B, distance="l1"):
    instance_A = np.squeeze(instance_A)
    instance_B = np.squeeze(instance_B)

    if distance == "l1": # Manhattan
        return np.sum(np.abs(instance_A - instance_B))
    elif distance == "l2": # Euclidean
        return np.sqrt(np.sum((instance_A - instance_B)**2))

def calculate_sparsity(instance_A, instance_B):
    instance_A = np.squeeze(instance_A)
    instance_B = np.squeeze(instance_B)
    length = instance_A.shape[0]
    diff_count = np.count_nonzero(instance_A != instance_B)
    return 1 - ((diff_count) / length)

def predict_outlier_with_isolation_forest(if_model, cf):
    # print(type(cf), cf.shape)
    cf = np.squeeze(cf)
    cf = cf.reshape((1,-1))
    y_pred = if_model.predict(cf)
    return 1 if y_pred[0] == 1 else 0


def write_data_json(dataset_name, 
                    classifier_name, 
                    closeness_l1, 
                    closeness_l2, 
                    sparsity, 
                    isolation_predict, 
                    random_seed, 
                    instance_id, 
                    timegan_id, 
                    information_gain, 
                    start, 
                    length, 
                    from_id, 
                    from_class):
    filename = experiment_result_path_individual
    # read json, if there is no content in this json, initialize it then.
    if os.path.exists(filename) and os.path.getsize(filename) > 0:
        with open(filename, 'r') as f:
            data = json.load(f)
    else:
        data = []  # initialize an empty container
    # new data entry
    new_record = {
        "Dataset": dataset_name,
        "Classifier_name": classifier_name,
        "To-be-explained_instance_id": instance_id,
        "Counterfactual_id":timegan_id,
        "Random_seed": random_seed,
        "Closeness": {
            "Closeness_L1": closeness_l1,
            "Closeness_L2": closeness_l2,
        },
        "Sparsity": sparsity,
        "isolation_predict": isolation_predict,
        "Shapelet_infomation":{
            "Information_gain":information_gain,
            "Start":start,
            "Length":length,
            "From_Instance (id)":from_id,
            "Class":from_class
        }
    }
    # append data to json file
    data.append(new_record)
    # save
    with open(filename, 'w') as f:
        json.dump(data, f)

def write_aggregate_data_json(dataset_name, 
                              classifier_name, 
                              avg_closeness_l1, 
                              avg_closeness_l2, 
                              avg_sparsity, 
                              avg_outliers, 
                              random_seed, 
                              instance_id,
                              cf_count,
                              path):
    
    filename = path
    # read json, if there is no content in this json, initialize it then.
    if os.path.exists(filename) and os.path.getsize(filename) > 0:
        with open(filename, 'r') as f:
            data = json.load(f)
    else:
        data = []  # initialize an empty container
    # new data entry
    new_record = {
        "Dataset": dataset_name,
        "Classifier_name": classifier_name,
        "To-be-explained_instance_id": instance_id,
        "Number of counterfactuals generated": cf_count,
        "Random_seed": random_seed,
        "Average_Closeness": {
            "Average_Closeness_L1": avg_closeness_l1,
            "Average_Closeness_L2": avg_closeness_l2,
        },
        "Average_Sparsity": avg_sparsity,
        "Average_Outliers": avg_outliers,
    }
    # append data to json file
    data.append(new_record)
    # save
    with open(filename, 'w') as f:
        json.dump(data, f)

def train_isolation_forest(X_train, random_seed):
    from sklearn.ensemble import IsolationForest
    if_model = IsolationForest(random_state=random_seed).fit(X_train)
    return if_model


# class AlibiExperiment():

#     def __init__(self, classifier, shape):
#         self.classifier = classifier
#         self.abili_cfi_model  = Counterfactual(
#             classifier,
#             shape,
#             # pred_proba,
#             target_proba=alibi_parameters["target_proba"], 
#             tol=alibi_parameters["tol"],
#             target_class=alibi_parameters["target_class"], 
#             max_iter=alibi_parameters["max_iter"], 
#             lam_init=alibi_parameters["lam_init"],
#             max_lam_steps=alibi_parameters["max_lam_steps"], 
#             learning_rate_init=alibi_parameters["learning_rate_init"])
        
#     def get_explanation(self, to_be_explained_instance, probas):
#         return self.abili_cfi_model.explain(to_be_explained_instance, probas)

    # def calculate_closeness_l1(self, cf, original):
    #     return
    
    # def calculate_closeness_l2(self, cf, original):
    #     return
    
    # def calculate_sparsity(self, cf, original):
    #     return
    
    # def calculate_plausity(self, X_train, instance, random_seed):
    #     train_isolation_forest(X_train, random_seed)
    #     return
    