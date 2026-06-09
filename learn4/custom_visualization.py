import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

print("=" * 60)
print("综合练习：自定义数据集可视化")
print("=" * 60)

# 设置中文
plt.rcParams['font.sans-serif'] = ['SimHei','Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 创建销售数据集
np.random.seed(42)
dates = pd.date_range('2024-01-01', periods=90, freq='D')
products = ['产品A', '产品B', '产品C', '产品D']
regions = ['北区', '南区', '东区', '西区']

data = []
for date in dates:
    for product in products:
        for region in regions:
            sales = np.random.randint(100, 1000)
            data.append([date, product, region, sales])

df = pd.DataFrame(data, columns=['日期', '产品', '区域', '销售额'])
print(f"数据集形状: {df.shape}")
print(f"前5行:\n{df.head()}")

# 创建高级仪表板
fig = plt.figure(figsize=(20, 12))
fig.suptitle('销售数据分析仪表板', fontsize=20, fontweight='bold')

# 1. 总销售额趋势
plt.subplot(2, 3, 1)
daily_sales = df.groupby('日期')['销售额'].sum()
plt.plot(daily_sales.index, daily_sales.values, linewidth=2)
plt.xlabel('日期')
plt.ylabel('销售额')
plt.title('日销售额趋势')
plt.xticks(rotation=45)

# 2. 产品销售额占比
plt.subplot(2, 3, 2)
product_sales = df.groupby('产品')['销售额'].sum()
plt.pie(product_sales.values, labels=product_sales.index, autopct='%1.1f%%')
plt.title('产品销售额占比')

# 3. 区域销售额对比
plt.subplot(2, 3, 3)
region_sales = df.groupby('区域')['销售额'].sum()
sns.barplot(x=region_sales.index, y=region_sales.values, palette='viridis')
plt.title('区域销售额对比')
plt.xlabel('区域')
plt.ylabel('总销售额')

# 4. 热力图：产品×区域
plt.subplot(2, 3, 4)
pivot = df.pivot_table(values='销售额', index='产品', columns='区域', aggfunc='sum')
sns.heatmap(pivot, annot=True, fmt='.0f', cmap='YlOrRd')
plt.title('产品-区域销售热力图')

# 5. 销售额分布
plt.subplot(2, 3, 5)
sns.histplot(df['销售额'], bins=30, kde=True)
plt.xlabel('销售额')
plt.ylabel('频数')
plt.title('销售额分布')
plt.axvline(df['销售额'].mean(), color='red', linestyle='--', label='均值')
plt.legend()

# 6. 周销售模式
plt.subplot(2, 3, 6)
df['星期'] = df['日期'].dt.dayofweek
weekly_sales = df.groupby('星期')['销售额'].mean()
week_labels = ['周一', '周二', '周三', '周四', '周五', '周六', '周日']
plt.plot(week_labels, weekly_sales.values, marker='o', linewidth=2)
plt.xlabel('星期')
plt.ylabel('平均销售额')
plt.title('周销售模式')

plt.tight_layout()
plt.savefig(r'E:\python project\ai-learning\learn4\sales_dashboard.png', dpi=150)
plt.show()

print("\n销售仪表板已生成")