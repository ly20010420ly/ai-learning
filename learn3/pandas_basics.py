from operator import index

import pandas as pd
import numpy as np
from pandas.core.interchange import column

print('=' * 60)
print("一维数据:series")
print('=' * 60)

s1 = pd.Series([1,2,3,4,5,6])
print(f"\n从列表创建：\n{s1}")

s2 = pd.Series({'a':1,'b':2,'c':3,'d':4})
print(f"\n从字典创建:\n{s2}")

s3 = pd.Series([1,2,3],index=['a','b','c'])
print(f"\n指定索引:\n{s3}")

#属性
print(f"\n值:{s3.values}")
print(f"索引:{s3.index}")
print(f"形状:{s3.shape}")

#计算
s4 = pd.Series([1,2,3,4])
s5 = pd.Series([10,20,30,40])
print(f"加法:\n{s4 + s5}\n")
print(f"乘法:\n{s4 * s5}\n")
print(f"平方:\n{s4 ** 2}\n")

print('=' * 60)
print("二维数据表:DataFrame")
print('=' * 60)

data = {
    '姓名':['张三','李四','王五','赵六'],
    '年龄':[18,28,38,48],
    '城市':['北京', '上海', '广州', '深圳'],
    '工资':[8000,6000,4000,5000]
}
df = pd.DataFrame(data)
print(f"从字典创建:\n{df}\n")
print(df.shape)

#从numpy数组创建
arr = np.array([[1,2,3],[4,5,6],[7,8,9]])
df2 = pd.DataFrame(arr,columns=['a','b','c'])
print(f"\n从数组创建:\n{df2}")

#DataFrame的属性
print(f"\n形状:{df.shape}")
print(f'列名:{df.columns.tolist()}')
print(f'索引:{df.index.tolist()}')
print(f'数据类型:\n{df.dtypes}')

#查看数据
print(f'\n前两行:\n{df.head(2)}')
print(f'\n后两行:\n{df.tail(2)}')
print(f'\n统计信息:\n{df.describe()}')
print(f'\n基本信息\n:{df.info()}')

print('\n' + '=' * 60)
print("数据选择与筛选:DataFrame")
print('=' * 60)

df3 = pd.DataFrame({
    'A':[1,2,3,4,5,6],
    'B':[10,20,30,40,50,60],
    'C':[100,200,300,400,500,600],
},index = ['row1','row2','row3','row4','row5','row6'])

print(f'\n原数据：\n{df3}')
#选择列
print(f'选择单列:\n{ df3["A"] }')
print(f'选择多列:\n{ df3[["A","C"]] }')
#选择行
print(f"按位置选择：\n{df3.iloc[1:4]}\n")
print(f"按索引选择：\n{df3.loc[['row1']]}\n")
#条件筛选
print(f"筛选A>3的行：\n{df3[df3['A']>3]}\n")
print(f"多条件筛选：\n{df3[(df3['A']>2) & (df3['B']<45)]}\n")
#使用qurey筛选
print(f"用query筛选:\n{df3.query('A >= 3 & B <= 40')}")
#用inis筛选
print(f"用isin帅选A在[2,4]的行:\n{df3[df3['A'].isin([2,4])]}")

print('\n' + '=' * 60)
print("数据操作")
print('=' * 60)
#添加新列
df3['D'] = df3['A']+df3['B']
print(f'添加新列D=A+B:\n{df3}\n')
#删除列
df3_dropeed = df3.drop(['D'],axis=1)
print(f'删除列D:\n{df3_dropeed}\n')
#修改列名
df3_renamed = df3.rename(columns={'A':'A_new','B':'B_new'})
print(f'修改A,C的列名:\n{df3_renamed}')
#排序
print(f'按A列降序排序:\n{df3.sort_values("A",ascending=False)}')

print('\n' + '=' * 60)
print("缺失值处理")
print('=' * 60)

df4 = pd.DataFrame({
    'A':[1,2,3,None,5],
    'B':[10,20,None,None,50],
    'C':[100,200,300,400,500]
})
print(f'原始数据:\n{df4}\n')
#缺失值统计
print(f'确实值统计:\n{df4.isnull().sum()}')
print(f'缺失值总数:{df4.isnull().sum().sum()}')
print(f'删除含缺失值的行:\n{df4.dropna()}')
#填充缺失行
print(f'用0填充:\n{df4.fillna(0)}')
print(f'用A的平均值填充:\n{df4.fillna(df4["A"].mean())}')
print(f"用前向填充:\n{df4.ffill()}")

#判断是否有缺失值
print(f'是否有缺失值:\n{df4.isnull().any().any()}')