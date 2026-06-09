import seaborn as ans
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

print("=" * 60)
print("seaborn 统计可视化")
print("=" * 60)

ans.set_style('whitegrid')
ans.set_palette('husl')

plt.rcParams['font.sans-serif'] = ['SimHei','Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

#加载内置数据集
tips = ans.load_dataset('tips')
iris = ans.load_dataset('iris')
print(f'tips数据集形状:{tips.shape}')
print(f'iris数据集形状：{iris.shape}')

#关系图
fig = plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
ans.scatterplot(data=tips,x='total_bill',y='tip',hue='time',size='size')
plt.title('小额消费VS总消费')
plt.subplot(1,2,2)
ans.scatterplot(data=iris,x='sepal_length',y='sepal_width',hue='species')
plt.title('花萼长度VS花萼宽度')
plt.tight_layout()
plt.savefig(r'E:\python project\ai-learning\learn4\relplot.png',dpi=150)
plt.show()

#分布图
fig,axes = plt.subplots(2,2,figsize=(12,10))
#直方图
ans.histplot(tips['total_bill'],bins=50,kde=True,ax=axes[0,0])
axes[0,0].set_title('总消费分布')
#KDE图
ans.kdeplot(data=tips,x='total_bill',hue='sex',ax=axes[0,1])
axes[0,1].set_title('不同性别消费分布图')
#箱线图
ans.boxplot(data=tips,x='day',y='total_bill',ax=axes[1,0])
axes[1,0].set_title('不同天数的消费分布')
#小提琴图
ans.violinplot(data=tips,x='day',y='total_bill',ax=axes[1,1])
axes[1,1].set_title('小提琴图')

plt.tight_layout()
plt.savefig(r'E:\python project\ai-learning\learn4\distributions.png',dpi=150)
plt.show()

#分类图
fig,axes = plt.subplots(1,3,figsize=(15,5))
#条形图
ans.barplot(data=tips,x='day',y='tip',hue='sex',ax=axes[0])
axes[0].set_title('不同天的消费')
#计数图
ans.countplot(data=tips,x='day',hue='smoker',ax=axes[1])
axes[1].set_title('不同天的吸烟者数量')
#点图
ans.pointplot(data=tips,x='time',y='total_bill',hue='sex',ax=axes[2])
axes[2].set_title('不同时段的消费')

plt.tight_layout()
plt.savefig(r'E:\python project\ai-learning\learn4\categorical.png',dpi=150)
plt.show()

# 5. 矩阵图

# 相关性热力图
numeric_cols = ['total_bill', 'tip', 'size']  # 或者用更自动的方式
corr = tips[numeric_cols].corr()
plt.figure(figsize=(8, 6))
ans.heatmap(corr, annot=True, cmap='coolwarm', center=0,
            square=True, linewidths=1, fmt='.2f')
plt.title('相关性热力图')
plt.savefig(r'E:\python project\ai-learning\learn4\categorical.png', dpi=150)
plt.show()

# 成对关系图（只取前100行避免太慢）
ans.pairplot(iris.head(100), hue='species', diag_kind='kde')
plt.savefig(r'E:\python project\ai-learning\learn4\categorical.png', dpi=150)
plt.show()