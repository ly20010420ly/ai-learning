import torch
import matplotlib.pyplot as plt

print("=" * 60)
print("PyTorch 自动求导（Autograd）教程")
print("=" * 60)

# 设置中文
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

#基本自动求导
#创建需要梯度的张量
x = torch.tensor(2.0, requires_grad=True)
print(f"x = {x}, requires_grad={x.requires_grad}")
#定义函数
y = x ** 2
print(f"y = x^2 = {y}")
#反向传播计算梯度
y.backward()
print(f"dy/dx = {x.grad}")

print("=" * 60)
print("多变量函数求导")
x = torch.tensor([1.0,2.0,3.0],requires_grad=True)
print(f"x = {x}")
#定义函数
f = torch.sum(x ** 2)
print(f"f = sum(x**2) = {f}")
#反向传播
f.backward()
print(f"梯度 dy/dx = {x.grad}")

print("=" * 60)
print("矩阵求导")
A = torch.randn(3,3,requires_grad=True)
x = torch.randn(3,requires_grad=True)
b = torch.randn(3,requires_grad=True)
#线性变换 y = A @ x+b
y = A @ x + b
print(f"A的形状:{A.shape}")
print(f"x的形状:{x.shape}")
print(f"b的形状:{b.shape}")
print(f"y的形状:{y.shape}")
#计算标量损失
loss = torch.sum(y ** 2)
loss.backward()
print(f"A梯度形状:{A.grad.shape}")
print(f"x梯度形状:{x.grad.shape}")
print(f"b梯度形状:{b.grad.shape}")

#停止梯度计算
print("=" * 60)
print("停止梯度计算")

x = torch.tensor(2.0,requires_grad=True)
y = x ** 2
#方法一：detach()
z = y.detach()
print(f"y.requires_grag:{y.requires_grad}")
print(f"z.requires_grad:{z.requires_grad}")
#方法二：with torch.no_grad
with torch.no_grad():
    w = x ** 3
    print(f"w.requires_grad:{w.requires_grad}")


#梯度清零
print("=" * 60)
print("梯度清零")
x = torch.tensor(2.0,requires_grad=True)
y = x ** 2
y.backward()
print(f"第一次反向传播后:{x.grad}")
#再次反向传播会累计梯度
y = x ** 2
y.backward()
print(f"再次反向传播后（累加:{x.grad}）")
#梯度清零
x.grad.zero_()
print(f"清零后:{x.grad}")


#实战：梯度下降优化
print("=" * 60)
print("实战：使用梯度下降找到函数最小值")

# 目标函数 f(x) = x^2 + 2*x + 1
# 最小值在 x = -1

def objective_function(x):
    return x ** 2 + x * 2 + 1
#初始化变量
x = torch.tensor(5.0, requires_grad=True)
learning_rate = 0.1
iterations = 50
#记录优化过程
x_history = []
loss_history = []
print("开始优化")
for i in range(iterations):
    #前向传播
    loss = objective_function(x)
    #反向传播
    loss.backward()
    #更新参数
    with torch.no_grad():
        x -= learning_rate * x.grad
    #记录
    x_history.append(x.item())
    loss_history.append(loss.item())
    #清零梯度
    x.grad.zero_()
    if (i+1) %10 == 0:
        print(f"迭代{i+1}:x = {x.item():.4f},loss = {loss.item():.4f}")
print(f"\n最终结果：x = {x.item():.4f},理论最小值点 = -1.0")
#可视化优化过程
fig,axes = plt.subplots(1,2,figsize=(12,5))
# 损失下降曲线
axes[0].plot(loss_history)
axes[0].set_xlabel('Iteration')
axes[0].set_ylabel('Loss')
axes[0].set_title('Loss下降曲线')
axes[0].grid(True, alpha=0.3)
# x值变化
axes[1].plot(x_history)
axes[1].axhline(y=-1, color='r', linestyle='--', label='理论最小值')
axes[1].set_xlabel('Iteration')
axes[1].set_ylabel('x value')
axes[1].set_title('参数x的优化过程')
axes[1].legend()
axes[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(r'E:\python project\ai-learning\learn6\gradient_descent_optimization.png', dpi=150)
plt.show()
print("\n自动求导教程完成")

