import numpy as np
print(f"numpy版本{np.__version__}")

print("=" * 50)
print("创建数组的多种方法")
print("=" * 50)

list_arr = np.array([1,2,3,4,5])
print(f"用列表创建矩阵")

zero_arr = np.zeros((3,4))
print(f"3*4的全零矩阵\n{zero_arr}")

one_arr = np.ones((2,3))
print(f"2*3的全一矩阵\n{one_arr}")

eye_arr = np.eye(4)
print(f"4*4的单位矩阵\n{eye_arr}")

arange1 = np.arange(0,10,2)
print(f'arange:{arange1}')
linspace_arr = np.linspace(0,1,5)
print(f'linspace(0,1,5):{linspace_arr}')

rand_arr = np.random.rand(2,3)
print(f"2*3 0-1均匀分布的随机矩阵\n{rand_arr}")

randn_arr = np.random.randn(2,3)
print(f"2*3 标准正态分布的随机矩阵\n{randn_arr}")

print("=" * 50)
print(f'数组属性')
print("=" * 50)

arr = np.array([[1,2,3],[4,5,6]])
print(f'数组：\n{arr}')
print(f"形状：{arr.shape}")
print(f"元素个数：{arr.size}")
print(f"维度：{arr.ndim}")
print(f"数据类型:{arr.dtype}")
print(f"每个元素字节数:{arr.itemsize}")

print("=" * 50)
print("数学运算")
print("=" * 50)

a = np.array([1,2,3,4])
b = np.array([1,2,3,4])

print(f"a + b = {a+b}")
print(f"a - b = {a-b}")
print(f"a * b = {a*b}")
print(f"a / b = {a/b}")
print(f"a ** b = {a**b}")

#矩阵乘法
A = np.array([[1,2],[3,4]])
B = np.array([[5,6],[7,8]])
print(f"矩阵乘法：\n{np.dot(A,B)}")
print(f"矩阵乘法的另一种写法：\n{A & B}")

arr1 = np.array([[1,2,3],[4,5,6]])
print(f"数组：\n{arr1}")
print(f"求和：{np.sum(arr1)}")
print(f"按行求和:{np.sum(arr1,axis=1)}")
print(f"按列求和：{np.sum(arr1,axis=0)}")
print(f"均值：{np.mean(arr1)}")
print(f"标准差：{np.std(arr1)}")
print(f"最大值：{np.max(arr1)}")
print(f"最小值：{np.min(arr1)}")
print(f"每行最大值位置：{np.argmax(arr1,axis=1)}")

#广播机制
print(f"每行加[1,2,3]:\n{arr1 + [1,2,3]}")
print(f"每个元素*2：\n{arr1 * 2}")

