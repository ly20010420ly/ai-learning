import pandas as pd
import numpy as np
import time

print("\n" + "=" * 60)
print("pandas高级操作")
print("=" * 60)

#apply函数
df = pd.DataFrame({
    'A':[1,2,3,4],
    'B':[10,20,30,40],
})

def my_fun(x):
    return x*2 +1

df['C'] = df['A'].apply(my_fun)
print(f"应用函数后：\n{df}")

#lambda表达式
df['D'] = df['B'].apply(lambda x: x ** 0.5)
print(f'开方运算：\n{df}')

print("\n" + "=" * 60)
print("向量化操作（性能对比）")
print("=" * 60)

#创建大数据集
large_df = pd.DataFrame({
    'x': np.random.randn(1000000),
    'y':np.random.randn(1000000)
})
#使用循环
start = time.time()
result_loop = []
for i in range(len(large_df)):
    result_loop.append(large_df.loc[i,'x']+large_df.loc[i,'y'])
loop_time = time.time() - start
#使用向量化
start = time.time()
result_vector = large_df['x'] + large_df['y']
vector_time = time.time() - start

print(f"循环耗时：{loop_time:.4f}秒")
print(f"向量化耗时：{vector_time:.4f}秒")
print(f"加速比：{loop_time/vector_time:.1f}倍")

print("\n" + "=" * 60)
print("数据透视表")
print("=" * 60)

sales = pd.DataFrame({
    '日期':pd.date_range('2024-01-01',periods=100,freq='D'),
    '产品':np.random.choice(['A','B','C'],100),
    '销售额':np.random.randint(100,1000,100),
    '区域':np.random.choice(['北区','南区','东区'],100)
})

#按月统计
sales['月份'] = sales['日期'].dt.month
pivot = pd.pivot_table(sales,
                       values='销售额',
                       index='区域',
                       columns='产品',
                       aggfunc='mean')
print(f'透视表：\n{pivot}')

print("\n" + "=" * 60)
print("时间序列")
print("=" * 60)

ts = pd.Series(np.random.randn(100),
               index=pd.date_range('2024-01-01',periods=100))
print(f'时间序列：\n{ts.head()}')
print(f'按月重采样：\n{ts.resample("ME").mean()}')