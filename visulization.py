import numpy as np
import matplotlib.pyplot as plt
# from pyts.datasets import load_gunpoint
# from pyts.transformation import ShapeletTransform
from process_data import *
from shapelets_transform import get_st
import config

# Get dataset
X_train, y_train, X_test, y_test = process_ucr_dataset("Wafer")

import matplotlib.pyplot as plt
import numpy as np

def plot_time_series(cf_series, orig_series, start, length):
    # transform data to meet matplot requirement
    cf_series = np.squeeze(cf_series)
    orig_series = np.squeeze(orig_series)

    plt.figure()
    # counterfactual instance
    plt.plot(cf_series, color='green', label='cf', linewidth=2, alpha=0.5)
    # plt.plot(range(start, start + length), cf_series[start: start + length], color='red', label='changed')

    # to-be-explained instance
    plt.plot(orig_series, color='blue', label='to-be-explained', linewidth=1,)
    plt.legend()
    plt.show()


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