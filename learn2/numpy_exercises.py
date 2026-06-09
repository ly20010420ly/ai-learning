import numpy as np

print("=" * 50)
print("numpu练习题")
print("=" * 50)

#创建一个5*5的随机矩阵并做归一化
print("\n创建一个5*5的随机矩阵并归一化到(0,1)")
arr = np.random.rand(5,5)
arr_normalized = ((arr-arr.min()) / (arr.max() - arr.min()))
# print(f"归一化后的矩阵：\n{arr_normalized}")
print(f"原矩阵的最大值：{arr.max():.3f},最小值{arr.min():.3f}")
print(f"归一化后矩阵的最大值：{arr_normalized.max():.3f},最小值{arr_normalized.min():.3f}")

#矩阵乘法
print("\n创建两个矩阵并且计算其乘积")
A = np.random.rand(2,3)
B = np.random.rand(3,5)
C = A @ B
print(A)
print(f"A的形状：{A.shape}")
print(f"B的形状：{B.shape}")
print(f"C的形状：{C.shape}")

#统计练习
print("\n生成100个随机数，并计算统计量")
D = np.random.rand(100)
print(f"平均值：{np.mean(D)}")
print((f"标准差：{np.std(D)}"))
print(f"最大值：{np.max(D)}")
print(f"最小值：{np.min(D)}")
print(f"中位数：{np.median(D)}")

# 条件筛选
print("生成10个0-100的整数并取出大于50的数")
arr1 = np.random.randint(0,100,10)
print(f"原数组：{arr1}")
print(f"大于50的数组：{arr1[ arr1>50 ]}")
print(f"大于50的个数：{sum(arr1>50)}")




