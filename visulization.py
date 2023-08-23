# from pyts.datasets import load_gunpoint
# from pyts.transformation import ShapeletTransform
import numpy as np
import matplotlib.pyplot as plt
from process_data import *
import seaborn as sns
import config

def plot_save_time_series(cf_series, orig_series, start, length, dataset_name, classifier_name, instance_id, random_seed, timegan_id, sp_idx, is_plot=False, is_save=False):
    # transform data to meet matplot requirement
    cf_series = np.squeeze(cf_series)
    orig_series = np.squeeze(orig_series)
    # sns.set_palette("pastel")

    plt.figure(figsize=(12, 6))
    plt.plot(orig_series, color='c', label='To-be-explained', linewidth=1.5, )
    plt.plot(cf_series, color='darkorange', label='Counterfactual', linewidth=2, alpha=0.8, linestyle='-')
    
    # plt.yscale('log')
    plt.legend()
    plt.tight_layout()
    if is_save:
        plt.savefig('Figures/' + dataset_name + "/" + classifier_name + "/" + str(sp_idx) + "_" + str(timegan_id) + "_" + str(instance_id) + "_" + str(random_seed) + '.pdf', format="pdf")
        print("Saved.")
    if is_plot: plt.show()

def show_dataset_visually(X_train, y_train, dataset_name):
    # Get index of A/B
    indices_A = [i for i, x in enumerate(y_train) if x == '1']
    indices_B = [i for i, x in enumerate(y_train) if x == '-1']
    plt.figure(figsize=(12, 6)) 

    count = 0
    # Plot A (Positive)
    for i in indices_A:
        # if count > len(indices_A) / 10: break
        plt.plot(X_train.iloc[i][0], label='Positive', color='darkorange', alpha=1)
        count += 1
    count = 0
    # Plot B (Negative)
    for i in indices_B:
        # if count > len(indices_B) / 10: break
        plt.plot(X_train.iloc[i][0], label='Negative', color="cornflowerblue", alpha=0.5)
        count += 1
   
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    # plt.title('Data distribution of')
    plt.savefig(dataset_name + '_classes.pdf', format="pdf")
    # plt.show()

def plot_bar_curve(bar_values,
                   curve_values,
             title,
             dataset_names=["ECG200", "ECG200", "FordA", "FordA", "Wafer", "Wafer", "MoteStrain", "MoteStrain"], 
             method_names=["mlxtend", "Time-CF", "mlxtend", "Time-CF", "mlxtend", "Time-CF", "mlxtend", "Time-CF"],
             bar_value_name="Manhattan distance (L1-norm)"):
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd
    data = {'Dataset': dataset_names,
            'Method': method_names,
            bar_value_name: bar_values}
    df = pd.DataFrame(data)
    sns.barplot(x='Dataset', y=bar_value_name, hue='Method', data=df)
    sns.lineplot(x='Dataset', y='Line', data=df, sort=False, color='red')
    plt.title(title)
    plt.savefig("eps_storage/" + bar_value_name + ".pdf", format='pdf')
    plt.show()

def plot_heatmap_missing_chart(data):
    import pandas as pd 
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(['red', 'blue'])
    data = {'KNN': [1, 1, 1, 1],
            'CNN': [1, 1, 0, 1],
            'DrCIF': [1, 1, 1, 1],
            'Catch22': [1, 1, 1, 1]}
    df = pd.DataFrame(data, index=['ECG200', 'FordA', 'Wafer', 'MoteStrain'])
    mask = df.isnull()
    sns.heatmap(df, mask=~mask, cmap=cmap, cbar=False)
    sns.heatmap(df, mask=mask, cmap=cmap, cbar=False)
    plt.savefig("eps_storage/" + "time_cf_missing.pdf", format='pdf')
    plt.show()

def plot_heatmap(data, file_name):
    import seaborn as sns
    import pandas as pd
    df = pd.DataFrame(data, index=['ECG200', 'FordA', 'Wafer', 'MoteStrain'])
    ax = sns.heatmap(df, annot=True, cmap=sns.cubehelix_palette(as_cmap=True))
    ax.set_title("Omission: " + file_name) 
    plt.savefig("eps_storage/" + file_name + ".pdf", format='pdf')
    plt.show()


def plot_heatmaps(data1, data2, data3):
    df_timecf = pd.DataFrame(data1, index=['ECG200', 'FordA', 'Wafer', 'MoteStrain'])
    df_ng = pd.DataFrame(data2, index=['ECG200', 'FordA', 'Wafer', 'MoteStrain'])
    df_mlxtend = pd.DataFrame(data3, index=['ECG200', 'FordA', 'Wafer', 'MoteStrain'])
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5), gridspec_kw={'width_ratios': [1, 1, 1.2]}, sharey=True)
    
    sns.heatmap(df_mlxtend, ax=ax1, cbar=False, cmap=sns.cubehelix_palette(as_cmap=True), annot=True)
    ax1.set_xlabel('mlxtend')
    sns.heatmap(df_timecf, ax=ax2, cbar=False, cmap=sns.cubehelix_palette(as_cmap=True), annot=True)
    ax2.set_xlabel('TimeCF')
    sns.heatmap(df_ng, ax=ax3, cbar=True, cmap=sns.cubehelix_palette(as_cmap=True), annot=True, vmin=0, vmax=1)
    ax3.set_xlabel('Native-Guide')

    ax1.set_aspect('equal', adjustable='box')
    ax2.set_aspect('equal', adjustable='box')
    ax3.set_aspect('equal', adjustable='box')
    
    fig.suptitle("Omission", x=0.5, y=0.96)  
    # plt.subplots_adjust(left=0.1, right=0.9)
    fig.tight_layout(pad=1)
    plt.savefig("eps_storage/Omission.pdf", format='pdf', bbox_inches='tight', pad_inches=0)
    plt.show()


def plot_curve():
    data = {
        'Dataset': ['ECG200', 'ECG200', 'ECG200', 
                    'FordA', 'FordA', 'FordA', 
                    'Wafer', 'Wafer', 'Wafer',
                    'MoteStrain', 'MoteStrain', 'MoteStrain'],
        'CF Method': ['mlxtend', 'Time-CF', 'Native-Guide', 
                    'mlxtend', 'Time-CF', 'Native-Guide', 
                    'mlxtend', 'Time-CF', 'Native-Guide',
                    'mlxtend', 'Time-CF', 'Native-Guide'],
        'Sparsity': 
        [0, 0.603, 0.122, 
        0, 0.879, 0.0157,
        0, 0.96, 0.072,
        0, 0.889, 0.0625]
    }

    df = pd.DataFrame(data)

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x='Dataset', y='Sparsity', hue='CF Method', marker="o")
    plt.title('Sparsity of different CF methods on different datasets')
    plt.savefig("eps_storage/" + "sparsity.pdf", format='pdf')
    plt.show()


def plot_bar(values,
             title,
             dataset_names=["ECG200", "ECG200", "ECG200", "FordA", "FordA", "FordA", "Wafer", "Wafer", "Wafer", "MoteStrain", "MoteStrain", "MoteStrain"], 
             method_names=["mlxtend", "Time-CF", "Native-Guide", "mlxtend", "Time-CF", "Native-Guide", "mlxtend", "Time-CF", "Native-Guide", "mlxtend", "Time-CF", "Native-Guide"],
             value_name="Manhattan distance (L1-norm)"):
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd
    data = {'Dataset': dataset_names,
            'Method': method_names,
            value_name: values}
    df = pd.DataFrame(data)
    # sns.set_palette("Paired")
    # palette = ["pink", "orange", "purple"]
    ax = sns.barplot(x='Dataset', y=value_name, hue='Method', data=df, )
    plt.title(title)
    for p in ax.patches:
        if p.get_height() == 0.002:
            ax.annotate("0%", (p.get_x() + p.get_width() / 2., 0), 
                        ha='center', va='center', 
                        xytext=(0, 5), textcoords='offset points', 
                        color="darkgray")
    plt.savefig("eps_storage/" + value_name + ".pdf", format='pdf')
    plt.show()

def plott_save_time_series(cf_series, orig_series, dataset_name, classifier_name, instance_id, random_seed, timegan_id, sp_idx, is_plot=False, is_save=False):
    # transform data to meet matplot requirement
    cf_series = np.squeeze(cf_series)
    orig_series = np.squeeze(orig_series)
    
    # Find overlapping indices
    overlapping_idx = np.where(cf_series == orig_series)[0]
    
    # Create a mask for non-overlapping indices in cf_series
    non_overlapping_mask = np.ones(cf_series.shape, dtype=bool)
    non_overlapping_mask[overlapping_idx] = False
    
    # Create new cf_series for non-overlapping parts
    non_overlap_cf_series = cf_series.astype(float).copy()
    non_overlap_cf_series[overlapping_idx] = np.nan
    
    plt.figure(figsize=(12, 6))
    
    # Plot the original series
    plt.plot(orig_series, color='blue', label='To-be-explained', linewidth=1)
    
    # Plot the counterfactual series where it doesn't overlap with the original series
    plt.plot(non_overlap_cf_series, color='green', label='cf (mlxtend)', linewidth=2, alpha=0.5)
    
    # Uncomment the next line if you want to set the y-axis to a logarithmic scale
    # plt.yscale('log')
    
    plt.legend()
    
    if is_save:
        plt.savefig('Figures/' + dataset_name + "/" + classifier_name + "/" + str(sp_idx) + "_" + str(timegan_id) + "_" + str(instance_id) + "_" + str(random_seed) + '.pdf', format="pdf")
    
    if is_plot:
        plt.show()

# Example usage
# cf_series = np.array([1, 2, 3, 4, 5, 7, 2, 1])
# orig_series = np.array([1, 2, 3, 6, 8, 5, 2, 1])
# plot_save_time_series(cf_series, orig_series, 0, 0, 'dataset_name', 'classifier_name', 1, 42, 1, 1, is_plot=True)


"""
Closeness:
"""
# plot_bar(title = "Closeness", 
#          value_name="Euclidean distance (L2-norm)",  
#          values=[10.9, 3.9, 5.3, 31.7, 11.6, 37, 12.8, 4.2, 7.9, 9.7, 3.0, 8.9])
# closeness - l1: "Manhattan distance (L1-norm)"
# values=[75.4, 17.7, 30.3, 573.4, 81.2, 555.1, 111.6, 9.7, 45.0, 55.3, 8.8, 43.3])
# closeness - l2: "Euclidean distance (L2-norm)"
# values=[10.9, 3.9, 5.3, 31.7, 11.6, 37, 12.8, 4.2, 7.9, 9.7, 3.0, 8.9]

"""
Sparsity:
"""

# plot_curve()

"""Omission"""
# omission: Time-CF
# data_timecf = {'KNN': [0, 0, 1, 0.33],
#             'CNN': [0, 0.17, 1, 0],
#             'DrCIF': [0.33, 0, 0.5, 0],
#             'Catch22': [0.17, 0, 0.83, 0]}
# # omission: Native-Guide
# data_ng = {'KNN': [0.5, 0.3, 0.5, 0.5],
#             'CNN': [0.7, 0.5, 0.5, 0.63],
#             'DrCIF': [0.57, 0.6, 0.5, 0.57],
#             'Catch22': [0.6, 0.53, 0.5, 0.6]}
# # omission: mlxtend
# data_mlxtend = {'KNN': [0.33, 0.5, 0.5, 0.5],
#             'CNN': [0.17, 0.67, 0.5, 0.67],
#             'DrCIF': [0, 0.33, 0.17, 0.17],
#             'Catch22': [0.17, 0.33, 0.17, 0]}

# plot_heatmaps(data_mlxtend, data_ng, data_timecf)

"""
Plausibitly:
"""
# plot_bar(
#     title = "Plausibility", 
#     value_name="Outliers (%)", 
    # values=[0.2725, 0.4775, 0.255, 0.002, 0.002, 0.255, 0.355, 0.33, 0.4, 0.002, 0.12875, 0.07375])
# values=[0.2725, 0.25625, 0.255, 0.002, 0.002, 0.255, 0.355, 0.33, 0.4, 0.002, 0.0625, 0.07375])




