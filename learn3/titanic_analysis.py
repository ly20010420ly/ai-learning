import numpy as np
import pandas as pd
import  matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
import os

from pandas.core.interchange.from_dataframe import primitive_column_to_ndarray

print('=' *60)
print("Titanic 数据分析实战")
print('=' *60)

print('加载数据...')
df = pd.read_csv('data/train.csv')

print(f'数据形状：{df.shape}')
print(f'前五行：\n{df.head()}\n')

print('=' *60)
print("数据探索")
print('=' *60)

print(f"数据类型：\n{df.dtypes}")
print(f'统计信息：\n{df.describe()}')
print(f'缺失值统计：\n{df.isnull()}')

print('=' *60)
print("数据清洗")
print('=' *60)

#处理年龄缺失值（中位数）
df['Age'].fillna(df['Age'].median(),inplace=True)
print(f'年龄缺失值填充后：\n{df["Age"].isnull().sum()}')
#处理船舱号缺失值（'Unknown'）
df['Cabin'].fillna('Unknown',inplace=True)
print(f'船舱号缺失值填充后：\n{df["Cabin"].isnull().sum()}')
#处理登船港口缺失值（众数）
df['Embarked'].fillna(df['Embarked'].mode()[0],inplace=True)
print(f'登船港口缺失值填充后：\n{df["Embarked"].isnull().sum()}')
#删除乘客ID和名字（分析时不需要）
df.drop(['PassengerId','Name','Ticket'],axis=1,inplace=True)

print(f'清洗后数据形状;{df.shape}')
print(f'清洗数据后缺失值统计:\n{df.isnull().sum().sum()}')

print('=' *60)
print("特征工程")
print('=' *60)

df['FamilySize'] = df['SibSp'] + df['Parch']
print('添加家庭大小特征')

#年龄分组
df['AgeGroup'] = pd.cut(df['Age'],
                        bins = [0,12,18,35,60,100],
                        labels=['儿童','青少年','青年','中年','老年'])
print('年龄分组完成')
#票价分组
df['FareGroup'] = pd.cut(df['Fare'],4,labels=['低价','中低价','中高价','高价'])
print('票价分组完成')
print(f'分组过后数据：{df.head()}')

print('=' *60)
print("数据分析结果")
print('=' *60)

#不同舱位的存活率
class_survival = df.groupby('Pclass')['Survived'].mean()
print(f"\n舱位存活率：\n{class_survival}")
#不同年龄的存活率
sex_survival = df.groupby('Sex')['Survived'].mean()
print(f"\n性别存活率：\n{sex_survival}")
#不同年龄的存活率
age_survival = df.groupby('AgeGroup')['Survived'].mean()
print(f"\n年龄组存活率：\n{age_survival}")
# 不同家庭大小的生存率
family_survival = df.groupby('FamilySize')['Survived'].mean()
print(f"\n家庭大小生存率:\n{family_survival}")

#数据可视化
print("\n6. 生成可视化图表...")
rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 中文字体
rcParams['axes.unicode_minus'] = False            # 解决负号显示问题
# 子图1：舱位与生存率
plt.subplot(2, 3, 1)
class_survival.plot(kind='bar', color=['red', 'blue', 'green'])
plt.title('不同舱位生存率')
plt.xlabel('舱位')
plt.ylabel('生存率')
plt.xticks(rotation=0)

# 子图2：性别与生存率
plt.subplot(2, 3, 2)
sex_survival.plot(kind='bar', color=['pink', 'lightblue'])
plt.title('不同性别生存率')
plt.xlabel('性别')
plt.ylabel('生存率')
plt.xticks(rotation=0)

# 子图3：年龄分布
plt.subplot(2, 3, 3)
df[df['Survived']==1]['Age'].hist(alpha=0.7, label='幸存', bins=20)
df[df['Survived']==0]['Age'].hist(alpha=0.7, label='遇难', bins=20)
plt.title('年龄分布对比')
plt.xlabel('年龄')
plt.ylabel('人数')
plt.legend()

# 子图4：年龄组生存率
plt.subplot(2, 3, 4)
age_survival.plot(kind='bar', color='skyblue')
plt.title('不同年龄组生存率')
plt.xlabel('年龄组')
plt.ylabel('生存率')
plt.xticks(rotation=45)

# 子图5：家庭大小与生存率
plt.subplot(2, 3, 5)
family_survival.plot(kind='line', marker='o')
plt.title('家庭大小对生存率的影响')
plt.xlabel('家庭成员数')
plt.ylabel('生存率')

# 子图6：票价与生存率
plt.subplot(2, 3, 6)
ax6 = plt.gca()  # 获取当前子图的 Axes
df.boxplot(column='Fare', by='Survived', ax=ax6)
plt.title('票价分布对比')
plt.suptitle('')  # 去掉 boxplot 默认生成的副标题
plt.xlabel('生存状态 (0=遇难, 1=幸存)')
plt.ylabel('票价')

plt.tight_layout()
plt.savefig('data/titanic_analysis.png', dpi=150)
plt.show()

print("\n可视化图表已保存为 titanic_analysis.png")

# 7. 输出分析报告
print("\n" + "=" * 60)
print("分析报告摘要")
print("=" * 60)

print("""
关键发现：
1. 头等舱存活率最高（约63%），三等舱最低（约24%）
2. 女性存活率（约74%）远高于男性（约19%）
3. 儿童存活率较高
4. 家庭大小适中（2-4人）存活率更高
5. 票价越高，存活率越高

建议：
- 舱位是最重要的影响因素
- 性别差异显著，体现"妇女儿童优先"
- 家庭规模适中有利于互相照应
""")

# 保存清洗后的数据
df.to_csv('data/titanic_cleaned.csv', index=False)
print("\n清洗后的数据已保存为 titanic_cleaned.csv")

