import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# 从文件中读取数据
file_path = 'data.txt'
with open(file_path, 'r') as file:
    data = [int(line.strip()) for line in file]

# 排序数据
sorted_data = sorted(data)

# 创建主图
fig, ax = plt.subplots(figsize=(14, 7))

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
    ax.bar(i, sorted_data[i], color=color)

# 添加标题和轴标签
ax.set_title('Long Tail Distribution of Word Frequencies', fontsize=16)
ax.set_xlabel('Index', fontsize=14)
ax.set_ylabel('Frequency', fontsize=14)

# 添加区间统计信息
ax.text(third / 2, max(sorted_data) * 0.9, 'Low Frequency', fontsize=12, color=colors[0], ha='center')
ax.text(3 * third / 2, max(sorted_data) * 0.9, 'Medium Frequency', fontsize=12, color=colors[1], ha='center')
ax.text(5 * third / 2, max(sorted_data) * 0.9, 'High Frequency', fontsize=12, color=colors[2], ha='center')

# 创建嵌套子图
ax_inset = inset_axes(ax, width="30%", height="30%", loc='upper left', borderpad=3)

# 绘制子图中的分布
ax_inset.hist(sorted_data[:third], bins=20, color=colors[0])
ax_inset.set_title('Low Frequency Distribution', fontsize=10)
ax_inset.set_xlabel('Frequency', fontsize=8)
ax_inset.set_ylabel('Count', fontsize=8)

# 保存图像为高清格式
plt.savefig('long_tail_distribution_with_inset.png', dpi=300, bbox_inches='tight')

# 显示图形
plt.show()