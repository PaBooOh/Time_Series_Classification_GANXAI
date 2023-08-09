# from sktime.datasets import load_UCR_UEA_dataset
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# from sktime.datasets import load_from_arff_to_dataframe
# from sktime.datatypes._panel._convert import from_nested_to_2d_array
from process_data import *
from shapelets_transform import*
from sklearn.metrics import classification_report
from visulization import *
import experiment
import numpy as np
import tensorflow as tf
import config


def main(dataset = config.dataset_name, classifier_name = config.classifier_name, random_seed = config.random_seed):
    """
    Configuration
    """
    dataset_name = dataset
    class_names = config.class_names
    classifier_name = classifier_name
    random_seed = random_seed

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
    
    print("(1) Training classifier ", classifier_name, " ", clf)
    print()

    if config.save_cls:
        clf.fit(X_train, y_train)
        clf.save(path=config.classifier_path)
    else:
        clf = clf.load_from_path(config.classifier_path + ".zip")
        print("Loading model at " + config.classifier_path + ".zip")
    y_pred = clf.predict(X_test)
    print("Report: ")
    print(classification_report(y_test, y_pred, target_names=class_names))
    print("Score: ", clf.score(X_test, y_test)) # score
    print()
    print("Training isolation forest model for anomaly detection ......")
    from sktime.datatypes._panel._convert import from_nested_to_2d_array
    X_train_float = from_nested_to_2d_array(X_train).to_numpy()
    # X_test_float = from_nested_to_2d_array(X_test).to_numpy()
    # X_train_float = np.array(X_train).reshape(-1, 1)
    if_model = experiment.train_isolation_forest(X_train_float, random_seed)

    # sum_normal_test = 0
    # for instance in X_test_float:
    #     y_pred = if_model.predict(instance.reshape(1, -1))
    #     sum_normal_test += 1
    #     # print()
    # print("Isolation forest test: ", sum_normal_test / 100.0)
    # print()
    # a
    # Specify a to-be-explained instance from test set.
    instance_id = config.instance_id # Test set instance
    to_be_explained_instance = X_test.iloc[instance_id][0].values.reshape(1, 1, seq_length)
    to_be_explained_instance_label_y = y_test[instance_id]
    to_be_explained_instance_predicted_y = clf.predict(to_be_explained_instance)[0]
    from copy import copy
    # original_instance = copy(to_be_explained_instance)
    target_y = 1 if int(to_be_explained_instance_predicted_y) == -1 else -1
    print("To-be-explained_instance_id: ", instance_id, ", Predicted: ", to_be_explained_instance_predicted_y, ", Truth: ", to_be_explained_instance_label_y, ", Target: ", target_y)
    print()

    test_if_y = 1 if target_y == -1 else 1
    """
    3 - Train Random Shapelet Transform model and get candidates
    """
    print("(2) Extracting Shapelet candidates based on sktime's Random Shapelet Transform......")
    print()
    st = get_shapelet_candidates_with_ST(min_len=3, 
                                        max_len=int(seq_length*0.5), 
                                        random_seed=random_seed, 
                                        time_limit=config.extract_time)


    if config.save_st:
        st.fit(X_train, y_train)
        st.save(path=config.st_saved_path)
    else:
        st = st.load_from_path(config.st_saved_path + ".zip")

    st.transform(X_train)


    sorted_shapelets = sorted(st.shapelets, key=lambda sp: sp[1]) # Sorted by length
    print("Statistics about all Shapelets extracted by Shapelet Transform: ")
    for idx, sp in enumerate(sorted_shapelets): # 0: information gain; 1: len; 2: start_pos; 4: instance id; 5: class; 
        # Show the extracted Shapelets candidates after Shapelets Transform (ST)
        print("Shapelet_Id: ", idx, ", Information gain: ", sp[0], ", Start: ", sp[2], ", Length: ", sp[1], ", From Instance id: ", sp[4], ", Class: ", sp[5])
    print()

    """
    4 - Get target (fake) instance's by TimeGAN
    """
    print("(3) Generating fake instances by TimeGAN ......")
    print()
    target_instances = get_data_by_class(X_train, y_train, target_y)
    fake_target_instances = generate_fake_sequences_by_TimeGAN(
        target_instances, 
        seq_length, 
        use_model=config.to_load_model, 
        path=config.model_saved_path)
    # print(fake_target_instances.shape)

    """
    5 - Generate counterfactual instances (plus: train a isolation forest model for anomaly detection)
    """

    print("(4) Generating counterfactual instance based on the generated fake instances and fake Shapelets ......")
    print()

    recorded_avg_closeness_l1 = 0
    recorded_avg_closeness_l2 = 0
    recorded_avg_sparsity = 0
    recorded_avg_outliers = 100
    recorded_cf_count = 0
    recorded_shapelet_id = -1

    search_continue = False
    sum_closeness_l1 = 0
    sum_closeness_l2 = 0
    sum_sparsity = 0
    sum_normal = 0 
    cf_count = 0 # count how many counterfactual instances are generated, also for Diversity measure

    # for-loop: iterate over generated target Shapelets (real) by Shapelet Transform based on the length (ascending order)
    for sp_idx, sp in enumerate(sorted_shapelets):
        # if search_continue:
        #     search_continue = False
        print("Start to find counterfactuals based on extracted Shapelet-Id: ", sp_idx, " ...")
            # break # retain the the case with minimum length
        # Look for the discriminative area of to-be-explained instance. That is, check if there is an extracted Shapelet having the same class as to-be-explained instance's class    
        if (int(sp[5]) == int(to_be_explained_instance_label_y)): 
            start_pos = sp[2] # sp[5] is class
            sp_length = sp[1]
            # Crop a specified intervals from target instances to get target shapelets for TimeGAN training
            fake_shapelets = crop_shapelets_from_dataset(
                X_train, 
                y_train, 
                length=sp_length,
                start_pos=start_pos)     
            # for-loop: iterate over generated target Shapelets (fake) by TimeGAN
            for timegan_idx, fake_shapelet in enumerate(fake_shapelets): # (80, 39 ,1)
                fake_shapelet = fake_shapelet.reshape(1, 1, sp_length)
                cf_instance = copy(to_be_explained_instance)
                cf_instance[:, :, start_pos:start_pos+sp_length] = fake_shapelet
                cf_predicted = clf.predict(cf_instance)[0]
                # Find out a cf with shortest length of replacement (Shapelets) and the rest of possible cf would be skipped.
                if (int(to_be_explained_instance_predicted_y) != int(cf_predicted)): # if cf is classified as target, meaning it is valid
                    search_continue = True
                    cf_count += 1 
                    # Experiment
                    ## Closeness: L1
                    closeness_l1 = experiment.calculate_closeness(cf_instance, to_be_explained_instance, "l1")
                    ## Closeness: L2
                    closeness_l2 = experiment.calculate_closeness(cf_instance, to_be_explained_instance, "l2")
                    ## Sparsity
                    sparsity = experiment.calculate_sparsity(cf_instance, to_be_explained_instance)
                    ## Plausibility: Isolation Forest
                    isolation_predict = experiment.predict_outlier_with_isolation_forest(if_model, cf_instance) # 1 normal, 0 anomaly otherwise.
                    sum_closeness_l1 += closeness_l1  # calculate l1
                    sum_closeness_l2 += closeness_l2  # calculate l2
                    sum_sparsity += sparsity # sparsity
                    sum_normal += isolation_predict
                    
                    print("A Counterfactual instance generated >>> Shapelet_Id: ", sp_idx, ", TimeGAN_id: ", timegan_idx)
                    # print("Information gain: ", sp[0], ", Start: ", sp[2], ", Length: ", sp[1], ", From Instance: ", sp[4], ", Class: ", sp[5])
                    # print("Closeness_l1: ", closeness_l1, ", Closeness_l2: ", closeness_l2, ", Sparsity: ", sparsity, ", Out-of-distribution: ", isolation_predict)
                    # print()
                    # 1. plot the comparison between to-be-explained and cf
                    # plot_save_time_series(
                    #     cf_instance, 
                    #     to_be_explained_instance, 
                    #     dataset_name=dataset_name, 
                    #     classifier_name=classifier_name, 
                    #     instance_id=instance_id, 
                    #     random_seed=random_seed,
                    #     timegan_id=timegan_idx,
                    #     sp_idx=sp_idx,
                    #     is_save=False,
                    #     is_plot=False)
                    
                    # 2. store experiment results for each cf based on a Shapelet
                    experiment.write_data_json(dataset_name, 
                                            classifier_name, 
                                            closeness_l1, 
                                            closeness_l2, 
                                            sparsity,
                                            isolation_predict,
                                            random_seed, 
                                            instance_id, 
                                            timegan_idx,
                                            sp[0], 
                                            sp[2], 
                                            sp[1], 
                                            sp[4], 
                                            sp[5],
                                            sp_idx)
                    # print("Generation stop successfully.")
                    # print()
            if cf_count != 0:
                avg_outliers = 1 - sum_normal / cf_count
                if (avg_outliers < recorded_avg_outliers):
                    recorded_avg_outliers = avg_outliers
                    recorded_avg_closeness_l1 = sum_closeness_l1 / cf_count
                    recorded_avg_closeness_l2 = sum_closeness_l2 / cf_count
                    recorded_avg_sparsity = sum_sparsity / cf_count
                    recorded_cf_count = cf_count
                    recorded_shapelet_id = sp_idx
                    if avg_outliers == 0.0:
                        break
        # 3. store aggregate (average) experiment results for all cfs based on a Shapelet
        if cf_count == 0:
            print("Shapelet " + str(sp_idx), " is not able to be used to find any counterfactual instances ...")
            print()
        else:
            print("Shapelet Id: " + str(sp_idx), " >>> avg_Closeness_l1: ", sum_closeness_l1/cf_count, ", avg_Closeness_l2: ", sum_closeness_l2/cf_count, ", avg_Sparsity: ", sum_sparsity/cf_count, ", avg_outliers ", 1 - sum_normal / cf_count)
    experiment.write_aggregate_data_json(
        dataset_name, 
        classifier_name, 
        recorded_avg_closeness_l1, 
        recorded_avg_closeness_l2, 
        recorded_avg_sparsity, 
        recorded_avg_outliers, 
        random_seed,
        instance_id,
        recorded_cf_count,
        recorded_shapelet_id,
        path=config.experiment_result_path_aggregate)

# experiment once
# main()

# or experiment on the whole dataset
for dataset in ["Wafer"]:
    for classifier in ["KNN", "CNN", "DrCIF", "Catch22"]:
        for seed in [111, 222, 333]:   
            main(dataset, classifier, seed)