# (Hyper) parameters

shapelet_len_ratio = 0.1
shapelet_len_ratios = [0.1, 0.2, 0.3, 0.4]

## General
instance_id = 33 
classifier_name = "DrCIF"
random_seed = 111
dataset_name = "FordA"
class_names = ['1', '-1']
experiment_result_path_individual = 'experiment_data.json'
experiment_result_path_aggregate = 'experiment_data_aggregate.json'

## TimeGAN
train_steps = 5000
model_saved_path = "TimeGAN_Models/" + dataset_name + "/" + dataset_name + "_" + str(train_steps) + "_" + str(random_seed) + ".pkl"
to_save_model = True
to_load_model = False
timegan_parameters = {
    "batch_size":4,
    "learning_rate":5e-4,
    "noise_dim":32,
    "layers_dim":64,
    "hidden_dim":24,
    "gamma":1
}
## Generation model


