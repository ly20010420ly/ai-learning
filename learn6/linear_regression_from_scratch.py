import torch
import  matplotlib.pyplot as plt
import numpy as np

# 设置中文
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("从零实现线性回归")
print("=" * 60)

# 1. 生成数据
print("\n1. 生成数据")

torch.manual_seed(42)

#真实参数
true_w = 2.5
true_b = 1.8

#生产数据
X = torch.randn(100,1)
y = true_w * X + true_b + torch.randn(100, 1) * 0.5

print(f"真实权重: {true_w}")
print(f"真实偏置: {true_b}")
print(f"数据形状: X={X.shape}, y={y.shape}")

# 2. 初始化参数
print("\n2. 初始化参数")

# 随机初始化
w = torch.randn(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)

print(f"初始权重: {w.item():.4f}")
print(f"初始偏置: {b.item():.4f}")

# 3. 定义模型和损失函数
def model(X, w, b):
    return X @ w + b

def loss_fn(y_pred, y_true):
    return torch.mean((y_pred - y_true) ** 2)

# 4. 训练
print("\n3. 训练模型")

learning_rate = 0.02
epochs = 500
loss_history = []
w_history = []
b_history = []

for epoch in range(epochs):
    # 前向传播
    y_pred = model(X, w, b)
    loss = loss_fn(y_pred, y)

    # 记录
    loss_history.append(loss.item())
    w_history.append(w.item())
    b_history.append(b.item())

    # 反向传播
    loss.backward()

    # 更新参数
    with torch.no_grad():
        w -= learning_rate * w.grad
        b -= learning_rate * b.grad

    # 清零梯度
    w.grad.zero_()
    b.grad.zero_()

    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch + 1}: loss={loss.item():.4f}, w={w.item():.4f}, b={b.item():.4f}")

# 5. 结果分析
print("\n4. 结果分析")
print(f"最终权重: {w.item():.4f} (真实值: {true_w})")
print(f"最终偏置: {b.item():.4f} (真实值: {true_b})")

# 6. 可视化
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 损失曲线
axes[0, 0].plot(loss_history)
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].set_title('损失下降曲线')
axes[0, 0].grid(True, alpha=0.3)

# 权重变化
axes[0, 1].plot(w_history, label='权重', color='blue')
axes[0, 1].axhline(y=true_w, color='r', linestyle='--', label='真实值')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Weight')
axes[0, 1].set_title('权重优化过程')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 偏置变化
axes[1, 0].plot(b_history, label='偏置', color='green')
axes[1, 0].axhline(y=true_b, color='r', linestyle='--', label='真实值')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Bias')
axes[1, 0].set_title('偏置优化过程')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# 拟合效果
with torch.no_grad():
    y_pred_final = model(X, w, b)

axes[1, 1].scatter(X.numpy(), y.numpy(), alpha=0.6, label='真实数据')
axes[1, 1].plot(X.numpy(), y_pred_final.numpy(), 'r-', linewidth=2, label='拟合直线')
axes[1, 1].set_xlabel('X')
axes[1, 1].set_ylabel('y')
axes[1, 1].set_title('线性回归拟合结果')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(r'E:\python project\ai-learning\learn6\linear_regression_result.png', dpi=150)
plt.show()


print("\n 线性回归从零实现完成")