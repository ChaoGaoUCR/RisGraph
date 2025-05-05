import matplotlib.pyplot as plt

# X 轴：HDN
hdn = [1, 2, 4, 8, 16]

# Y 轴两组数据
projection_accuracy = [7.43, 62.83, 82.19, 93.6, 95.8]
without_projection_accuracy = [5.32, 55.0, 62.94, 65.2, 65.26]

# 创建图形
plt.figure(figsize=(8, 5))
plt.plot(hdn, projection_accuracy, marker='o', label='Projection Accuracy')
plt.plot(hdn, without_projection_accuracy, marker='s', label='Without Projection Accuracy')

# 设置标签
plt.xlabel("HDN")
plt.ylabel("Accuracy (%)")
plt.grid(True)
plt.legend(loc="lower right")
plt.title("")  # 无标题

# 保存为 PDF
plt.tight_layout()
plt.savefig("hdn_vs_accuracy.pdf")
plt.close()

print("✅ Saved as 'hdn_vs_accuracy.pdf'")
