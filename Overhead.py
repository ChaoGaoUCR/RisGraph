import matplotlib.pyplot as plt
import numpy as np

# 数据定义
datasets = ["SX", "Wiki", "TTW", "Wen", "OR", "DL"]
sampling = [3.96, 0.97, 2.94, 2.95, 2.42, 2.64]
construction = [15.17, 66.46, 15.20, 9.09, 37.43, 29.34]
diskio = [34.78, 9.07, 25.52, 35.22, 33.56, 32.95]
collection = [32.21, 10.13, 48.38, 42.53, 13.36, 22.79]
query = [13.89, 13.37, 7.96, 10.21, 13.23, 12.29]

# 设置柱状图参数
ind = np.arange(len(datasets))
width = 0.6

# 绘图
fig, ax = plt.subplots(figsize=(8, 5))
p1 = ax.bar(ind, sampling, width)
p2 = ax.bar(ind, construction, width, bottom=sampling)
p3 = ax.bar(ind, diskio, width, bottom=np.array(sampling)+np.array(construction))
p4 = ax.bar(ind, collection, width, bottom=np.array(sampling)+np.array(construction)+np.array(diskio))
p5 = ax.bar(ind, query, width, bottom=np.array(sampling)+np.array(construction)+np.array(diskio)+np.array(collection))

# 设置标签
ax.set_xticks(ind)
ax.set_xticklabels(datasets)
ax.set_ylabel('Time Percentage (%)')
ax.set_xlabel('')
ax.set_title('')  # 无标题
ax.legend((p1[0], p2[0], p3[0], p4[0], p5[0]), ('Sampling', 'Construction', 'DiskIO', 'Collection', 'Query'))

# 保存为 PDF
plt.tight_layout()
plt.savefig("Overhead.pdf")
plt.close()
print("✅ PDF saved as 'stacked_bar_chart.pdf'")
