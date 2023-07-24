# from pyts.datasets import load_gunpoint
# from pyts.transformation import ShapeletTransform
import numpy as np
import matplotlib.pyplot as plt
from process_data import *
import config

def plot_save_time_series(cf_series, orig_series, dataset_name, classifier_name, instance_id, random_seed, timegan_id, sp_idx, is_plot=False, is_save=False):
    # transform data to meet matplot requirement
    cf_series = np.squeeze(cf_series)
    orig_series = np.squeeze(orig_series)


    plt.figure(figsize=(12, 6))
    # plit counterfactual instance
    
    # plt.plot(range(start, start + length), cf_series[start: start + length], color='red', label='changed')
    
    plt.plot(cf_series, color='green', label='cf', linewidth=2, alpha=0.5)
    plt.plot(orig_series, color='blue', label='to-be-explained', linewidth=1,)
    # plt.yscale('log')
    plt.legend()
    if is_save:
        plt.savefig('Figures/' + dataset_name + "/" + classifier_name + "/" + str(sp_idx) + "_" + str(timegan_id) + "_" + str(instance_id) + "_" + str(random_seed) + '.png')
    if is_plot: plt.show()

def show_dataset_visually(X_train, y_train, dataset_name):
    # Get index of A/B
    indices_A = [i for i, x in enumerate(y_train) if x == '1']
    indices_B = [i for i, x in enumerate(y_train) if x == '-1']
    plt.figure(figsize=(12, 6)) 

    count = 0
    # Plot A (Positive)
    for i in indices_A:
        if count > len(indices_A) / 10: break
        plt.plot(X_train.iloc[i][0], label='Positive', color='darkorange', alpha=0.8)
        count += 1
    count = 0
    # Plot B (Negative)
    for i in indices_B:
        if count > len(indices_B) / 10: break
        plt.plot(X_train.iloc[i][0], label='Negative', color="cornflowerblue", alpha=0.3)
        count += 1
   
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    # plt.title('Data distribution of')
    plt.savefig('Figures/' + dataset_name + '_classes.png')
    # plt.show()

# def plot_bar(dataset_names=["ECG200", "MoteStrain", "Wafer"], 
#              method_names=["mlxtend", "Time-CF", "mlxtend", "Time-CF", "mlxtend", "Time-CF"]):
#     # Category=["mlxtend", "Time-CF"], 
#     import seaborn as sns
#     import matplotlib.pyplot as plt
#     import pandas as pd
#     data = {'Dataset': dataset_names,
#             'Method': method_names,
#             'Value': [7, 5, 12, 10, 5, 7]}
#     # data = {'Dataset': ['Dataset 1', 'Dataset 1', 'Dataset 2', 'Dataset 2', 'Dataset 3', 'Dataset 3', 'Dataset 4', 'Dataset 4', 'Dataset 5', 'Dataset 5'],
#     #         'Method': ['Method AAA', 'Method B', 'Method A', 'Method B', 'Method A', 'Method B', 'Method A', 'Method B', 'Method A', 'Method B'],
#     #         'Value': [7, 5, 12, 10, 5, 7, 18, 15, 6, 8]}
#     df = pd.DataFrame(data)
#     sns.barplot(x='Dataset', y='Value', hue='Method', data=df)
#     plt.show()

# plot_bar()
