# from pyts.datasets import load_gunpoint
# from pyts.transformation import ShapeletTransform
import numpy as np
import matplotlib.pyplot as plt
from process_data import *
import seaborn as sns
import config

def plot_save_time_series(cf_series, orig_series, dataset_name, classifier_name, instance_id, random_seed, timegan_id, sp_idx, is_plot=False, is_save=True):
    # transform data to meet matplot requirement
    cf_series = np.squeeze(cf_series)
    orig_series = np.squeeze(orig_series)

    plt.figure(figsize=(12, 6))
    # plit counterfactual instance
    
    # plt.plot(range(start, start + length), cf_series[start: start + length], color='red', label='changed')
    
    plt.plot(cf_series, color='green', label='cf (mlxtend)', linewidth=2, alpha=0.5)
    plt.plot(orig_series, color='blue', label='To-be-explained', linewidth=1,)
    # plt.yscale('log')
    plt.legend()
    if is_save:
        plt.savefig('Figures/' + dataset_name + "/" + classifier_name + "/" + str(sp_idx) + "_" + str(timegan_id) + "_" + str(instance_id) + "_" + str(random_seed) + '.pdf', format="pdf")
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
    # for p in ax.patches:
    #     if p.get_height() == 0:
    #         ax.annotate("0%", (p.get_x() + p.get_width() / 2., 0), 
    #                     ha='center', va='center', 
    #                     xytext=(0, 5), textcoords='offset points', 
    #                     color="darkgray")
    plt.savefig("eps_storage/" + value_name + ".pdf", format='pdf')
    plt.show()


"""
Closeness:
"""
# plot_bar(title = "Closeness", 
#          value_name="Euclidean distance (L2-norm)",  
#          values=[10.9, 4.4, 5.3, 31.7, 11.6, 37, 12.8, 4.2, 7.9, 9.7, 3.9, 8.9])
# closeness - l1:
# values=[75.4, 21.9, 30.3, 573.4, 81.2, 555.1, 111.6, 9.7, 45.0, 55.3, 8.9, 43.3])
# closeness - l2:
# values=[10.9, 4.4, 5.3, 31.7, 11.6, 37, 12.8, 4.2, 7.9, 9.7, 3.9, 8.9]

"""
Sparsity:
"""

# plot_curve()
"""
Plausibitly:
"""
# plot_bar(
#     title = "Plausibility", 
#     value_name="Outliers (%)", 
#     values=[0.2725, 0.25625, 0.255, 0, 0, 0.255, 0.355, 0.33, 0.4, 0, 0.0625, 0.07375])
## time-cf-omission
# data = {'KNN': [0, 0, 1, 0],
#             'CNN': [0, 0.17, 1, 0],
#             'DrCIF': [0, 0, 0.33, 0],
#             'Catch22': [0, 0, 0.67, 0]}
# plot_heatmap(data, "Time-CF")

## time-cf-omission
# data = {'KNN': [0, 0, 1, 0],
#             'CNN': [0, 0.17, 1, 0],
#             'DrCIF': [0, 0, 0.33, 0],
#             'Catch22': [0, 0, 0.67, 0]}
## native-guide
# data = {'KNN': [0.5, 0.3, 0.5, 0.5],
#             'CNN': [0.7, 0.5, 0.5, 0.63],
#             'DrCIF': [0.57, 0.6, 0.5, 0.57],
#             'Catch22': [0.6, 0.53, 0.5, 0.6]}
## mlxtend
# data = {'KNN': [0.33, 0.5, 0.5, 0.5],
#             'CNN': [0.17, 0.67, 0.5, 0.67],
#             'DrCIF': [0, 0.33, 0.17, 0.17],
#             'Catch22': [0.17, 0.33, 0.17, 0]}


