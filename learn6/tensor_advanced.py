import torch

print("=" * 60)
print("张量高级操作")
print("=" * 60)

# 1. 张量拼接
print("\n1. 张量拼接")

a = torch.randn(2,3)
b = torch.randn(2,3)
print(f"a的形状:{a.shape}")
print(f"b的形状:{b.shape}")

#水平拼接（按列）
c = torch.cat([a,b],dim=0)
print(f"dim=0拼接后的形状:{c.shape}")
#垂直拼接（按行）
d = torch.cat([a,b],dim=1)
print(f"dim=1拼接后的形状:{d.shape}")


# 2. 张量堆叠
print("\n2. 张量堆叠")
stacked = torch.stack([a,b],dim=0)
print(f"stack dim=0形状: {stacked.shape}")
stacked2 = torch.stack([a,b],dim=1)
print(f"stack dim=1形状: {stacked2.shape}")

# 3. 张量分割
print("\n3. 张量分割")

tensor = torch.randn(6,4)
print(f"原张量形状: {tensor.shape}")
#按块分割
chunks = torch.chunk(tensor,2,dim=1)
print(f"分成2块，每块形状: {chunks[0].shape}")
#按大小分割
splits = torch.split(tensor,[2,4],dim=0)
print(f"分成[2,4]块，形状: {splits[0].shape}, {splits[1].shape}")


# 4. 数学函数
print("\n4. 数学函数")
x = torch.tensor([0.0,1.0,2.0,3.0])
print(f"x = {x}")

print(f"exp(x) = {torch.exp(x)}" )      #  x的e次方
print(f"log(x) = {torch.log(x+1)}")     #  ln(x)
print(f"sin(x) = {torch.sin(x)}")
print(f"sqrt(x) = {torch.sqrt(x)}")     #根号x
print(f"abs(x-1.5) = {torch.abs(x-1.5)}")   #绝对值
print(f"max(x) = {torch.max(x)}")
print(f"min(x) = {torch.min(x)}")
print(f"mean(x) = {torch.mean(x)}")
print(f"sum(x) = {torch.sum(x)}")


# 5. 比较操作
print("\n5. 比较操作")

a = torch.tensor([1,2,3,4])
b = torch.tensor([2,2,2,2])

print(f"a = {a}")
print(f"b = {b}")
print(f"a > b： {a > b}")
print(f"a = b: {a == b}")
print(f"torch.where(a > b,a,b:{torch.where(a > b,a,b)}") #取最大值


# 6. 随机数生成
print("\n6. 随机数生成")

#设置随机种子（保证可重复性）
torch.manual_seed(42)
#均匀分布
uniform = torch.rand(3,3)
print(f"均匀分布[0,1):\n{uniform}")
#正态分布
normal = torch.randn(3,3)
print(f"标准正态分布:\n{normal}")
#整数随机
integer = torch.randint(0,10,(3,3))
print(f"整数随机[0,10):\n{integer}")


# 7. 数据类型转换
print("\n7. 数据类型转换")

float_tensor = torch.tensor([1.0,2.0,3.0])
print(f"浮点型: {float_tensor.dtype}")
#转换为整数
int_tensor = float_tensor.long()
print(f"整数型：{int_tensor.dtype}")
#转换为float64
double_tensor = float_tensor.double()
print(f"双精度型：{double_tensor.dtype}")

#转换设备
if torch.cuda.is_available():
    cuda_tensor = float_tensor.cuda()
    print(f"CUDA设备：{cuda_tensor.device}")
    cpu_tensor = float_tensor.cpu()
    print(f"cpu设备:{cpu_tensor.device}")
print("\n 张量高级操作完成")