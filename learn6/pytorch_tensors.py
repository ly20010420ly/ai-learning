import torch
import numpy as np

print('=' * 60)
print("PyTorch 张量基础练习")
print('=' * 60)

#1.创建张量的多种方法
#从列表创建
tensor_from_list = torch.tensor([1,2,3,4,5,6])
print(f"从列表创建:\n{tensor_from_list}")
#从numpy数组创建
np_array = np.array([1,2,3,4,5,6])
tensor_from_numpy = torch.from_numpy(np_array)
print(f"从numpy创建:\n{tensor_from_numpy}")
#全零张量
zeros = torch.zeros(3,4)
print(f"3 * 4 的全0张量:\n{zeros}")
#全1张量
ones = torch.ones(2,3)
print(f"2*3的全1张量:\n{ones}")
#单位矩阵
eye = torch.eye(4)
print(f"4*4的单位矩阵:\n{eye}")
#随机张量
random_tensor = torch.rand(3,4)
print(f"3*4的随机矩阵:\n{random_tensor}")
#正态分布的随机张量
normal_tensor = torch.randn(3,5)
print(f"3*4的正态分布矩阵:\n{normal_tensor}")
#指定范围的张量
range_tensor = torch.arange(0,10,2)
print(f"arange(0,10,2):\n{range_tensor}")
#线性间隔张量
linspace_tensor = torch.linspace(0,1,5)
print(f"linspace(0,1,5):\n{linspace_tensor}")

print('=' * 60)
print("张量属性")
print('=' * 60)

tensor = torch.randn(3,4,5)
print(f"张量形状:\n{tensor.shape}")
print(f"张量维度数:\n{tensor.ndim}")
print(f"张量元素个数:\n{tensor.numel()}")
print(f"张量数据类型:\n{tensor.dtype}")
print(f"张量设备:{tensor.device}")
print(f"张量是否需要梯度:{tensor.requires_grad}")

print('=' * 60)
print("改变张量形状")
print('=' * 60)

tensor = torch.arange(12)
print(f"原张量：{tensor}")
print(f"原张量形状:{tensor.shape}")
#reshape
reshaped = tensor.reshape(3,4)
print(f"reshape(3,4):\n{reshaped}")
#view
viewed = tensor.view(4,3)
print(f"view(4,3):\n{viewed}")
#展平
flattened = tensor.flatten()
print(f"flatten():\n{flattened}")
#转置
matrix = torch.rand(3,4)
print(f"原矩阵形状:{matrix.shape}")
print(f"转置后的形状:{matrix.T.shape}")
#添加维度
unsqueezed = tensor.unsqueeze(0)
print(f"unsqueezed(0)形状:\n{unsqueezed.shape}")
#删除维度
squeezed = tensor.squeeze(0)
print(f"squeezed(0)形状:\n{squeezed.shape}")

print("=" * 60)
print("张量运算")

#基本算术运算
a = torch.tensor([1,2,3,4])
b = torch.tensor([5,6,7,8])
print(f"a = {a}")
print(f"b = {b}")
print(f"a + b = {a + b}")
print(f"a - b = {a - b}")
print(f"a * b = {a * b}")
print(f"a / b = {a / b}")
print(f"a ** 2 = {a ** 2}")

#矩阵乘法
A = torch.randn(3,4)
B = torch.randn(4,5)
C = torch.mm(A,B)
print(f"A形状:{A.shape},B形状:{B.shape}，C形状:{C.shape}")

#点积
vec1 = torch.tensor([1,2,3])
vec2 = torch.tensor([4,5,6])
dot_product = torch.dot(vec1,vec2)
print(f"点积:{dot_product}")

#广播机制
matrix = torch.randn(3,4)
vector = torch.tensor([1,2,3,4])

#索引和切片
print('=' * 60)
print("索引和切片")

tensor = torch.tensor([[1,2,3,4],
                       [5,6,7,8],
                       [9,10,11,12]])

print(f"原张量:\n{tensor}")

print(f"第一行:{tensor[0]}")
print(f"第一列:{tensor[:,0]}")
print(f"第一行第一列：{tensor[0,0]}")
print(f"前两行:\n{tensor[:2]}")
print(f"后两列:\n{tensor[:,-2:]}")
print(f"条件筛选：{tensor[tensor > 5]}")

#GPU支持
print('=' * 60)
print("GPU支持")
if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"GPU设备: {torch.cuda.get_device_name(0)}")

    # 将张量移到GPU
    tensor_gpu = tensor.to(device)
    print(f"张量所在设备: {tensor_gpu.device}")

    # 移回CPU
    tensor_cpu = tensor_gpu.cpu()
else:
    print("CUDA不可用，使用CPU")
    print("如需GPU加速，请安装CUDA版本的PyTorch")

print("\nPyTorch张量基础完成")
