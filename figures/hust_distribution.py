import matplotlib.pyplot as plt
import numpy as np

# 示例数据：替换为你的实际数据

# 从文件中读取数据
file_path = 'data.txt'
with open(file_path, 'r') as file:
    data = [int(line.strip()) for line in file]

# 排序数据
sorted_data = sorted(data)

# 设置Seaborn的风格
# sns.set(style="whitegrid")

# 创建柱状图
plt.figure(figsize=(14, 7))

# 定义颜色
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # 蓝色, 橙色, 绿色

# 计算区间
n = len(sorted_data)
third = n // 3

# 绘制柱状图
for i in range(n):
    if i < third:
        color = colors[0]
    elif i < 2 * third:
        color = colors[1]
    else:
        color = colors[2]
    plt.bar(i, sorted_data[i], color=color)

# 添加标题和轴标签
plt.title('Long Tail Distribution of Word Frequencies', fontsize=16)
plt.xlabel('Index', fontsize=14)
plt.ylabel('Frequency', fontsize=14)

# 添加区间统计信息
plt.text(third / 2, max(sorted_data) * 0.9, 'Low Frequency', fontsize=12, color=colors[0], ha='center')
plt.text(3 * third / 2, max(sorted_data) * 0.9, 'Medium Frequency', fontsize=12, color=colors[1], ha='center')
plt.text(5 * third / 2, max(sorted_data) * 0.9, 'High Frequency', fontsize=12, color=colors[2], ha='center')

# 保存图像为高清格式
plt.savefig('long_tail_distribution.png', dpi=300, bbox_inches='tight')

# 显示图形
plt.show()