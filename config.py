# (Hyper) parameters

shapelet_len_ratio = 0.1
shapelet_len_ratios = [0.1, 0.2, 0.3, 0.4]

## General
instance_id = 555
classifier_name = "KNN"
random_seed = 111
dataset_name = "FordA"
class_names = ['1', '-1']
experiment_result_path_individual = 'experiment_data.json'
experiment_result_path_aggregate = 'experiment_data_aggregate.json'
save_generated_cf_figure = True

## Random Shapelet Transform
extract_time = 5 # per minutes
st_saved_path = "ST_Models/" + dataset_name + "/" + dataset_name + "_" + str(random_seed)
save_st = False

## TimeGAN
train_steps = 5000
model_saved_path = "TimeGAN_Models/" + dataset_name + "/" + dataset_name + "_" + str(train_steps) + "_" + str(random_seed) + ".pkl"
to_load_model = True
timegan_parameters = {
    "batch_size":4,
    "learning_rate":5e-4,
    "noise_dim":32,
    "layers_dim":64,
    "hidden_dim":24,
    "gamma":1
}

## Classifiers
save_cls = True
classifier_path = "Classifiers_Models/" + classifier_name + "/" + classifier_name + "_" + dataset_name + "_" + str(random_seed)

## Comparative study: mlxtend
comparative_result_path_aggregate = 'comparative_study_mlxtend.json'
## Comparative study: Alibi

alibi_parameters = {
"target_proba": 1.0,
"tol": 0.01, # want counterfactuals with p(class)>0.99
"target_class": 'other', # any class other than 7 will do
"max_iter": 1000,
"lam_init": 1e-1,
"max_lam_steps": 10,
"learning_rate_init": 0.1,
}


