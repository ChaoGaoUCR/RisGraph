import matplotlib.pyplot as plt

# 数据
time_windows = ["4%", "8%", "12%", "16%", "20%", "24%", "28%", "32%", "36%", "40%"]
common_core = [93.50, 86.75, 78.53, 68.64, 59.21, 49.32, 38.88, 28.50, 19.89, 14.16]
union_core = [93.77, 88.32, 82.59, 77.26, 71.23, 65.29, 58.90, 53.40, 48.05, 42.74]
whole_core = [99.40, 99.29, 99.09, 98.58, 97.77, 96.69, 95.14, 93.03, 90.13, 86.64]

# 绘图
plt.figure(figsize=(10, 5))
plt.plot(time_windows, common_core, marker='o', label='Common Core Accuracy')
plt.plot(time_windows, union_core, marker='s', label='Union Core Accuracy')
plt.plot(time_windows, whole_core, marker='^', label='Whole Core Accuracy')

# 设置标签
plt.xlabel('Time window percentage')
plt.ylabel('Accuracy (%)')
plt.grid(True)
plt.legend(loc='lower left')
plt.title('')  # 无标题

# 导出 PDF
plt.tight_layout()
plt.savefig("UnionCore.pdf")
plt.close()

print("✅ Saved as 'accuracy_over_time_windows.pdf'")
