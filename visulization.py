import numpy as np
import matplotlib.pyplot as plt
# from pyts.datasets import load_gunpoint
# from pyts.transformation import ShapeletTransform
from process_data import *
from shapelets_transform import get_shapelet_candidates_with_ST
import config

def plot_time_series(cf_series, orig_series, start, length):
    # transform data to meet matplot requirement
    cf_series = np.squeeze(cf_series)
    orig_series = np.squeeze(orig_series)

    # cf_series_smooth = cf_series_pd.rolling(window=10).mean()


    plt.figure(figsize=(12, 6))
    # counterfactual instance
    plt.plot(cf_series, color='green', label='cf', linewidth=2, alpha=0.5)
    # plt.plot(range(start, start + length), cf_series[start: start + length], color='red', label='changed')

    # to-be-explained instance
    plt.plot(orig_series, color='blue', label='to-be-explained', linewidth=1,)
    # plt.yscale('log')
    plt.legend()
    plt.show()

def show_dataset_visually(X_train, y_train, dataset_name):
    # 获取A类和B类的索引
    indices_A = [i for i, x in enumerate(y_train) if x == '1']
    indices_B = [i for i, x in enumerate(y_train) if x == '-1']
    plt.figure(figsize=(12, 6))  # 创建大尺寸的图形

    count = 0
    # 画出A类的时间序列
    for i in indices_A:
        if count > len(indices_A) / 10: break
        plt.plot(X_train.iloc[i][0], label='Positive', color='darkorange', alpha=0.8)
        count += 1
    count = 0
    # 画出B类的时间序列
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



# ts1 = X_test.iloc[781][0].values.reshape(1, 1, 152)
# ts2 = X_test.iloc[782][0].values.reshape(1, 1, 152)
# plot_time_series(ts1, ts2, 20, 20)

# ts_length = get_time_series_length(X_test)
# print(X_train.iloc[95][0].shape)
# Shapelet transformation
# st = get_st(random=True)
# st.fit(X_train, y_train)
# X_new = st.transform(X)
# Index -- length: 1, start: 2, end : start+length+1
# for sp in st.shapelets:
#     print(sp[0], sp[1], sp[2], sp[4], sp[5])
# st = ShapeletTransform(window_sizes=config.shapelet_len_ratios,
#                        random_state=42, sort=True, n_shapelets=2)
# X_new = st.fit_transform(X_train, y_train)

# Plotting
# plt.figure(figsize=(24, 16))
# for i, sp in enumerate(st.shapelets):
#     idx, start, end = sp[4], sp[1], sp[2] + sp[1] + 1
#     plt.plot(X_train.iloc[idx][0], color='C{}'.format(i),
#              label='Sample {}'.format(idx))
#     plt.plot(np.arange(start, end), X_train.iloc[idx][0][start:end],
#              lw=5, color='C{}'.format(i))

# plt.xlabel('Time', fontsize=12)
# plt.title('The four most discriminative shapelets', fontsize=14)
# plt.legend(loc='best', fontsize=8)
# plt.show()