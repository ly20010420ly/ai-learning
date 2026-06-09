import pandas as pd
import numpy as np

print('=' * 60)
print('数据合并')
print('=' * 60)

df1 = pd.DataFrame({
    'id':[1,2,3,4],
    '姓名':['张三','李四','王五','赵六'],
    '部门':['销售','技术','销售','市场']
})
df2 = pd.DataFrame({
    'id':[1,2,3,5],
    '工资':[8000,12000,9500,10000],
    '入职年份':[2023,2025,2023,2020]
})
print(f'df1:\n{df1}\ndf2:\n{df2}')
inner_merge = pd.merge(df1,df2,on="id",how="inner")   #只连接都有的
print(f'内连接：\n{inner_merge}\n')
left_merge = pd.merge(df1,df2,on="id",how="left")     #以左边id准，右边没有的元素补None
print(f'左连接：\n{left_merge}\n')
right_merge = pd.merge(df1,df2,on="id",how="right")   #以右边id为准，左边没有的元素补None
print(f'右连接：\n{right_merge}\n')
outer_merge = pd.merge(df1,df2,on="id",how="outer")   #连接全部id，对方没有的补None
print(f'外连接：\n{outer_merge}\n')

# 拼接（垂直方向）
df3 = pd.DataFrame({
    'id': [6, 7],
    '姓名': ['小明', '小红'],
    '部门': ['技术', '市场']
})

concat_result = pd.concat([df1,df3],ignore_index=True)
print(f"\n垂直拼接:\n{concat_result}")