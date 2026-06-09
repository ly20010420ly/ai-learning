import matplotlib.pyplot as plt
import numpy as np

print("=" * 60)
print("matplotlib 基础教程")
print("=" * 60)

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

#折线图
x = np.linspace(0,10,100)
y1 = np.sin(x)
y2 = np.cos(x)
plt.figure(figsize=(10,6))
plt.plot(x,y1,'b-',linewidth=2,label='sin(x)')
plt.plot(x,y2,'r--',linewidth=2,label='cos(x)')
plt.xlabel('x')
plt.ylabel('y')
plt.title('正弦函数和余弦函数',fontsize=14)
plt.legend()
plt.grid(True,alpha=0.3)
plt.savefig(r'E:\python project\ai-learning\learn4\line_plot.png',dpi=150)
plt.show()

#散点图
np.random.seed(42)
x = np.random.randn(100)
y = 2 * x + 1 + np.random.randn(100) * 0.5
plt.figure(figsize=(8,6))
plt.scatter(x,y,alpha=0.6,c='blue',marker='o')
plt.xlabel('x')
plt.ylabel('y')
plt.title('散点图：带噪声的线性关系')
plt.grid(True,alpha=0.3)
plt.savefig(r'E:\python project\ai-learning\learn4\scatter_polt.png',dpi=150)
plt.show()

#柱状图
categories = ['A', 'B', 'C', 'D', 'E']
values = [23,45,56,78,32]

plt.figure(figsize=(8,6))
plt.bar(categories,values,color=['red','blue','green','orange','purple'])
plt.xlabel('类别')
plt.ylabel('数值')
plt.title('柱状图示例')
for i,v in enumerate(values):
    plt.text(i,v+1,str(v),ha='center')
plt.savefig(r'E:\python project\ai-learning\learn4\bar_polt.png',dpi=150)
plt.show()

#直方图
data = np.random.randn(1000)
plt.figure(figsize=(8,6))
plt.hist(data,bins=30,edgecolor='black',alpha=0.7)
plt.xlabel('数值')
plt.ylabel('频数')
plt.title('直方图:正态分布')
plt.grid(True,alpha=0.3)
plt.savefig(r'E:\python project\ai-learning\learn4\histogram.png',dpi=150)
plt.show()

#箱线图
data1 = np.random.randn(100)
data2 = np.random.randn(100) + 2
data3 = np.random.randn(100) - 1

plt.figure(figsize=(8,6))
plt.boxplot([data1,data2,data3],label=['组1','组2','组3'])
plt.xlabel('分组')
plt.ylabel('数值')
plt.title('箱线图对比')
plt.grid(True,alpha=0.3)
plt.savefig(r'E:\python project\ai-learning\learn4\box_plot.png',dpi=150)
plt.show()