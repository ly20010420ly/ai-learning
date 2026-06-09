import pandas as pd
import numpy as np

print('=' * 60)
print('分组聚合 groupby')
print('=' * 60)

#创建分组数据
sales_data = {
    '产品':['手机','电脑','手机','平板','电脑','手机','平板'],
    '销售额':[5000,8000,6000,3000,9000,5500,3500],
    '数量':[5,8,6,3,9,5,3],
    '城市':['北京','上海','广州','北京','上海','广州','北京']
}
df = pd.DataFrame(sales_data)
print(f"原始数据:\n{df}")
#单列分组
print(f'按产品聚合，计算销售额总和:\n{df.groupby("产品")["销售额"].sum()}')
print(f"按产品聚合，计算销售额平均值：\n{df.groupby('产品')['销售额'].mean()}")
#多列分组
print(f'按产品和城市分组：\n{df.groupby(["产品","城市"])["销售额"].sum()}')
#多个聚合函数
print(f'多种统计：\n{df.groupby("产品")["销售额"].agg(["sum","mean","std","count"])}')
#对所有数值列聚合
all = {
    '销售额':['sum','mean'],
    '数量' :['sum']
}
print(f"\n对全部数值列聚合:\n{df.groupby('产品').agg(all)}")
