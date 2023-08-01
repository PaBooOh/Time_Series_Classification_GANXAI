# from pyts.datasets import load_gunpoint
# from pyts.transformation import ShapeletTransform
import numpy as np
import matplotlib.pyplot as plt
from process_data import *
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

def plot_bar(values,
             title,
             dataset_names=["ECG200", "ECG200", "FordA", "FordA", "Wafer", "Wafer", "MoteStrain", "MoteStrain"], 
             method_names=["mlxtend", "Time-CF", "mlxtend", "Time-CF", "mlxtend", "Time-CF", "mlxtend", "Time-CF"],
             value_name="Manhattan distance (L1-norm)"):
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd
    data = {'Dataset': dataset_names,
            'Method': method_names,
            value_name: values}
    df = pd.DataFrame(data)
    sns.barplot(x='Dataset', y=value_name, hue='Method', data=df)
    plt.title(title)
    plt.savefig("eps_storage/" + value_name + ".pdf", format='pdf')
    plt.show()

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

def plot_heatmap_missing_chart():
    import seaborn as sns
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

def plot_heatmap_outliers():
    import seaborn as sns
    import pandas as pd
    data = {'KNN': [1, 0, 0.33, 0],
            'CNN': [0.92, 0, np.nan, np.nan],
            'DrCIF': [0.67, 0, 0.43, 0.14],
            'Catch22': [0.88, 0, 0.33, 0.05]}
    df = pd.DataFrame(data, index=['ECG200', 'FordA', 'Wafer', 'MoteStrain'])
    sns.heatmap(df, annot=True, cmap=sns.cubehelix_palette(as_cmap=True))
    plt.savefig("eps_storage/" + "mlxtend_outliers.pdf", format='pdf')
    plt.show()

# plot_heatmap_outliers()    
# plot_heatmap_missing_chart()
# show_dataset_visually()
# closeness-l1:
# plot_bar(title = "Closeness", value_name="Manhattan distance (L1-norm)",  values=[70.83, 8.8, 620.2, 48.45, 136.7, 13.1, 72.8, 9.1])
# closeness-l2:
# plot_bar(title = "Closeness", value_name="Euclidean distance (L2-norm)",  values=[10.6, 3.4, 33.4, 8.8, 14.6, 7.2, 12.6, 3.7])
# # sparsity:
# plot_bar(
#     title = "Plausibility", 
#     value_name="Outliers (%)", values=[0.87, 0.5, 0, 0, 0.36, 1, 0.06, 0.06])

# # outliers
# plot_bar(
#     title = "Plausibility", 
#     value_name="Outliers (%)", values=[0.87, 0.5, 0, 0, 0.36, 1, 0.06, 0.06])
## 1) time_cf
# data = {'KNN': [0.76, 0, 1, 0],
#             'CNN': [0.69, 0, np.nan, 0.26],
#             'DrCIF': [0.26, 0, 1, 0],
#             'Catch22': [0.46, 0, 1, 0]}

## 2) 
