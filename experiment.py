import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import UnivariateSpline

# 创建一个包含20个随机元素的时间序列
time_series = np.random.rand(100)

# 创建时间步长
time_steps = np.arange(1, 101)

# 使用UnivariateSpline进行平滑
spl = UnivariateSpline(time_steps, time_series, s=1.0)

# 生成新的时间步长以获得更高的解析度
time_steps_new = np.linspace(1, 100, 500)

# 画出第1-10步的时间序列
plt.plot(time_steps_new[0:250], spl(time_steps_new[0:250]), color='blue')

# 画出第10-15步的时间序列
plt.plot(time_steps_new[249:375], spl(time_steps_new[249:375]), color='green', alpha=0.5)

# 画出第15-20步的时间序列
plt.plot(time_steps_new[374:], spl(time_steps_new[374:]), color='blue')

plt.show()
