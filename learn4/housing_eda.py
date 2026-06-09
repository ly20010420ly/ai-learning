import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.datasets import fetch_california_housing
warnings.filterwarnings('ignore')

print("=" * 60)
print("波士顿房价数据完整EDA分析")
print("=" * 60)

# 设置中文
plt.rcParams['font.sans-serif'] = ['SimHei','Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

housing = fetch_california_housing()
df = pd.DataFrame(housing.data, columns=housing.feature_names)
df['MedHouseVal'] = housing.target
print("使用加州房价数据集")

print(f"数据形状: {df.shape}")
print(f"列名: {df.columns.tolist()}")
print(f"\n前5行:\n{df.head()}")

# 2. 数据概览
print("\n2. 数据概览")
print(f"\n数据类型:\n{df.dtypes}")
print(f"\n缺失值统计:\n{df.isnull().sum()}")
print(f"\n统计信息:\n{df.describe()}")

fig = plt.figure(figsize=(20, 16))
fig.suptitle('波士顿房价数据分析报告', fontsize=20, fontweight='bold')

# 子图1：房价分布
plt.subplot(3, 3, 1)
sns.histplot(df['MedHouseVal'], bins=50, kde=True, color='darkred')
plt.xlabel('房价（单位：10万美元）')
plt.ylabel('频数')
plt.title('房价分布直方图', fontsize=12)
plt.axvline(df['MedHouseVal'].mean(), color='red', linestyle='--', label=f'均值: {df["MedHouseVal"].mean():.2f}')
plt.axvline(df['MedHouseVal'].median(), color='green', linestyle='--', label=f'中位数: {df["MedHouseVal"].median():.2f}')
plt.legend()

# 子图2：收入分布
plt.subplot(3, 3, 2)
sns.histplot(df['MedInc'], bins=50, kde=True, color='blue')
plt.xlabel('收入（单位：10万美元）')
plt.ylabel('频数')
plt.title('收入分布', fontsize=12)

# 子图3：房价 vs 收入
plt.subplot(3, 3, 3)
sns.scatterplot(data=df, x='MedInc', y='MedHouseVal', alpha=0.5, s=20)
plt.xlabel('收入')
plt.ylabel('房价')
plt.title('房价与收入的关系', fontsize=12)
# 添加趋势线
z = np.polyfit(df['MedInc'], df['MedHouseVal'], 1)
p = np.poly1d(z)
plt.plot(df['MedInc'].sort_values(), p(df['MedInc'].sort_values()), 'r-', linewidth=2)

# 子图4：房价 vs 房龄
plt.subplot(3, 3, 4)
sns.scatterplot(data=df, x='HouseAge', y='MedHouseVal', alpha=0.5, s=20)
plt.xlabel('房龄')
plt.ylabel('房价')
plt.title('房价与房龄的关系', fontsize=12)

# 子图5：房价 vs 房间数
plt.subplot(3, 3, 5)
sns.scatterplot(data=df, x='AveRooms', y='MedHouseVal', alpha=0.5, s=20)
plt.xlabel('平均房间数')
plt.ylabel('房价')
plt.title('房价与房间数的关系', fontsize=12)

# 子图6：房价 vs 人口
plt.subplot(3, 3, 6)
sns.scatterplot(data=df, x='Population', y='MedHouseVal', alpha=0.5, s=20)
plt.xlabel('人口')
plt.ylabel('房价')
plt.title('房价与人口的关系', fontsize=12)

# 子图7：箱线图 - 收入分组的房价
plt.subplot(3, 3, 7)
df['IncomeGroup'] = pd.qcut(df['MedInc'], q=4, labels=['低收入', '中低收入', '中高收入', '高收入'])
sns.boxplot(data=df, x='IncomeGroup', y='MedHouseVal')
plt.xlabel('收入组')
plt.ylabel('房价')
plt.title('不同收入组的房价分布', fontsize=12)
plt.xticks(rotation=45)

# 子图8：相关性热力图
plt.subplot(3, 3, 8)
corr = df[['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'MedHouseVal']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, fmt='.2f', square=True)
plt.title('特征相关性热力图', fontsize=12)

# 子图9：地理分布
plt.subplot(3, 3, 9)
scatter = plt.scatter(df['Longitude'], df['Latitude'],
                      c=df['MedHouseVal'], cmap='hot',
                      alpha=0.6, s=10)
plt.colorbar(scatter, label='房价')
plt.xlabel('经度')
plt.ylabel('纬度')
plt.title('房价地理分布', fontsize=12)

plt.tight_layout()
plt.savefig(r'E:\python project\ai-learning\learn4\housing_eda_report.png', dpi=150)
plt.show()

# 4. 深入分析
print("\n4. 深入分析")

# 收入最高的10个区域
print("\n收入最高的10个区域:")
top_income = df.nlargest(10, 'MedInc')[['MedInc', 'MedHouseVal']]
print(top_income)

# 房价最高的10个区域
print("\n房价最高的10个区域:")
top_price = df.nlargest(10, 'MedHouseVal')[['MedHouseVal', 'MedInc']]
print(top_price)

# 分组统计
print("\n按收入分组统计:")
income_stats = df.groupby('IncomeGroup')[['MedHouseVal', 'MedInc']].agg(['mean', 'median', 'std'])
print(income_stats)

# 5. 高级可视化
print("\n5. 生成高级可视化图表")

# Joint plot
plt.figure(figsize=(10, 8))
joint = sns.jointplot(data=df, x='MedInc', y='MedHouseVal', kind='reg', height=8)
joint.fig.suptitle('收入与房价的联合分布', y=1.02)
plt.savefig(r'E:\python project\ai-learning\learn4\joint_plot.png', dpi=150)
plt.show()

# 多个变量的分布对比
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
features = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup']
for i, (ax, feature) in enumerate(zip(axes.flatten(), features)):
    sns.histplot(data=df, x=feature, kde=True, ax=ax)
    ax.set_title(f'{feature} 分布')
    ax.axvline(df[feature].mean(), color='red', linestyle='--')
plt.tight_layout()
plt.savefig(r'E:\python project\ai-learning\learn4\features_distribution.png', dpi=150)
plt.show()

# 6. 生成分析报告
print("\n" + "=" * 60)
print("分析报告摘要")
print("=" * 60)

report = """
房价数据分析报告 - 关键发现

1. 数据概况:
   - 总样本数: {n_samples}
   - 平均房价: {mean_price:.2f} (单位：10万美元)
   - 房价范围: [{min_price:.2f}, {max_price:.2f}]

2. 主要影响因素:
   - 收入与房价呈强正相关（相关系数: {inc_corr:.2f}）
   - 房间数与房价正相关（相关系数: {rooms_corr:.2f}）
   - 房龄与房价相关性较弱

3. 关键洞察:
   - 高收入区域的房价显著高于低收入区域
   - 房价在地理上呈现区域聚集性
   - 房价分布呈右偏态，说明存在少数高价区

4. 建议:
   - 收入是最重要的预测因子
   - 地理位置也是重要因素
   - 需要考虑特征之间的交互作用
"""

print(report.format(
    n_samples=len(df),
    mean_price=df['MedHouseVal'].mean(),
    min_price=df['MedHouseVal'].min(),
    max_price=df['MedHouseVal'].max(),
    inc_corr=df['MedInc'].corr(df['MedHouseVal']),
    rooms_corr=df['AveRooms'].corr(df['MedHouseVal'])
))

# 保存清洗后的数据
df.to_csv(r'E:\python project\ai-learning\learn4\housing_data_cleaned.csv', index=False)
print("\n数据已保存为 housing_data_cleaned.csv")
print("图表已保存: housing_eda_report.png, joint_plot.png, features_distribution.png")
