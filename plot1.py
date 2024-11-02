import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 指定文件路径
FileName = "D:\\桌面\\超算第一次作业\\矩阵数据\\Data10.txt"

# 加载数据
data = np.loadtxt(FileName)

# 创建网格
x, y = np.meshgrid(np.arange(data.shape[1]), np.arange(data.shape[0]))

# 绘制 3D 图像
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, y, data, cmap='viridis')

# 显示图像
plt.show()
