import matplotlib.pyplot as plt
import numpy as np

print("=" * 60)
print("matplotlib 高级技巧")
print("=" * 60)

# 设置中文
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 1. 子图布局
print("\n1. 多子图布局")
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle('多子图示例', fontsize=16)

x = np.linspace(0,10,100)
axes[0][0].plot(x,np.sin(x))
axes[0][0].set_title('sin(x)')
axes[0][0].grid(True)

axes[0, 1].scatter(np.random.randn(50), np.random.randn(50))
axes[0, 1].set_title('随机散点')

axes[0, 2].bar(['A', 'B', 'C'], [10, 20, 15])
axes[0, 2].set_title('柱状图')

data = np.random.randn(200)
axes[1, 0].hist(data, bins=20)
axes[1, 0].set_title('直方图')

axes[1, 1].boxplot([np.random.randn(50), np.random.randn(50) + 1])
axes[1, 1].set_title('箱线图')

sizes = [30, 25, 20, 15, 10]
labels = ['A', 'B', 'C', 'D', 'E']
axes[1, 2].pie(sizes, labels=labels, autopct='%1.1f%%')
axes[1, 2].set_title('饼图')

plt.tight_layout()
plt.savefig(r'E:\python project\ai-learning\learn4\subplots.png',dpi=150)
plt.show()



print("\n2. 自定义样式")
plt.figure(figsize=(10, 6))

# 不同线条样式
x = np.linspace(0, 10, 20)
plt.plot(x, x, 'r-', linewidth=2, label='实线')
plt.plot(x, x**0.5, 'g--', linewidth=2, label='虚线')
plt.plot(x, np.log(x+1), 'b:', linewidth=2, label='点线')
plt.plot(x, x**2, 'y-.', linewidth=2, label='点划线')

plt.xlabel('X轴', fontsize=12)
plt.ylabel('Y轴', fontsize=12)
plt.title('不同线条样式对比', fontsize=14)
plt.legend(loc='upper left')
plt.grid(True, alpha=0.3, linestyle='--')
plt.savefig(r'E:\python project\ai-learning\learn4\line_styles.png', dpi=150)
plt.show()

# 3. 颜色和标记
print("\n3. 颜色和标记")
plt.figure(figsize=(10, 6))

markers = ['o', 's', '^', 'D', '*']
colors = ['red', 'blue', 'green', 'orange', 'purple']

for i, (marker, color) in enumerate(zip(markers, colors)):
    x = np.linspace(i, i+5, 10)
    y = np.sin(x) + i
    plt.plot(x, y, marker=marker, color=color,
             markersize=8, linewidth=2,
             label=f'系列{i+1}')

plt.xlabel('X')
plt.ylabel('Y')
plt.title('不同颜色和标记')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(r'E:\python project\ai-learning\learn4\colors_markers.png', dpi=150)
plt.show()

# 4. 填充和误差条
print("\n4. 填充和误差条")
x = np.linspace(0, 10, 20)
y = np.sin(x)
error = 0.2 + 0.1 * np.random.randn(20)

plt.figure(figsize=(10, 6))
plt.plot(x, y, 'b-', linewidth=2, label='均值')
plt.fill_between(x, y - error, y + error, alpha=0.3, color='blue', label='误差范围')
plt.errorbar(x[::2], y[::2], yerr=error[::2], fmt='ro',
             markersize=6, capsize=3, label='数据点')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('带填充和误差条的曲线')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(r'E:\python project\ai-learning\learn4\fill_errorbar.png', dpi=150)
plt.show()