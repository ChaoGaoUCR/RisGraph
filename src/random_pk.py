import random

# 定义生成数字的数量和范围
num_count = 1000
min_val = 0
max_val = 13593032

# 打开文件，写入随机数，每个数字占一行
with open("wen.txt", "w") as file:
    for _ in range(num_count):
        number = random.randint(min_val, max_val)
        file.write(f"{number}\n")

print("随机数字已生成并保存为 wen.txt")
