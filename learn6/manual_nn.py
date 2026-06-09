import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# 设置中文
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("手写神经网络实现")
print("=" * 60)

# 1. 生成模拟数据
print("\n1. 生成模拟数据")

np.random.seed(42)
torch.manual_seed(42)

#生成非线性数据
X =  np.random.randn(200,1)
y = 2 * X ** 2 + 3*X + 1 + np.random.randn(200,1) *0.5

#转换为张量
X_tensor = torch.FloatTensor(X)
y_tensor = torch.FloatTensor(y)
print(f"X形状:{X_tensor.shape}")
print(f"y形状:{y_tensor.shape}")

# 2. 手动实现神经网络（不使用nn.Module）
print("\n2. 手动实现神经网络")

class ManualNN:
    def __init__(self,input_size,hidden_size,output_size):
        #初始化权重和偏置
        self.W1 = (torch.randn(input_size,hidden_size) * 0.1).requires_grad_()
        self.b1 = torch.zeros(hidden_size,requires_grad=True)
        self.W2 = (torch.randn(hidden_size,output_size) * 0.1).requires_grad_()
        self.b2 = torch.zeros(output_size,requires_grad=True)

    def forward(self,x):
        #第一层：线性变换 + ReLU激活
        self.z1 = x @ self.W1 + self.b1
        self.a1 = torch.relu(self.z1)
        #第二层：线性变换
        self.z2 = self.a1 @ self.W2 + self.b2
        return self.z2

    def parameters(self):
        return [self.W1,self.W2,self.b1,self.b2]

    def zero_grad(self):
        for param in self.parameters():
            if param.grad is not None:
                param.grad.zero_()

#创建模型
model_manual = ManualNN(1,10,1)
print("手动神经网络创建成功")
print(f"输入层: 1, 隐藏层: 10, 输出层: 1")

# 3. 训练循环
print("\n3. 训练模型")

learning_rate = 0.01
epochs = 1000
loss_history_manual = []

for epoch in range(epochs):
    #前向传播
    predictions = model_manual.forward(X_tensor)
    #计算损失(均方误差)
    loss = torch.mean((predictions - y_tensor) ** 2)
    loss_history_manual.append(loss.item())
    #反向传播
    loss.backward()
    #手动更新参数
    with torch.no_grad():
        for param in model_manual.parameters():
            param -= learning_rate * param.grad
    #清零梯度
    model_manual.zero_grad()
    if (epoch+1) % 200 == 0:
        print(f"epoch: {epoch+1}, loss: {loss.item():.4f}")


# 4. 使用PyTorch的nn.Module实现
print("\n4. 使用nn.Module实现神经网络")
class PyTorchNN(nn.Module):
    def __init__(self,input_size,hidden_size,output_size):
        super(PyTorchNN,self).__init__()
        self.fc1 = nn.Linear(input_size,hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size,output_size)

    def forward(self,x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

#创建模型
model_pytorch = PyTorchNN(1,10,1)
print("PyTorch NN模型创建成功")
print(f"模型结构:\n{model_pytorch}")

#定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.SGD(model_pytorch.parameters(),lr=learning_rate)

#训练PyTorch模型
loss_history_pytorch = []

for epoch in range(epochs):
    #前向传播
    predictions = model_pytorch(X_tensor)
    loss = criterion(predictions,y_tensor)
    loss_history_pytorch.append(loss.item())
    #反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 200 == 0:
        print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")

# 5. 可视化训练结果
print("\n5. 可视化训练结果")

fig,axes = plt.subplots(2,2,figsize=(14,10))

#损失曲线对比
# 损失曲线对比
axes[0, 0].plot(loss_history_manual, label='Manual NN', alpha=0.7)
axes[0, 0].plot(loss_history_pytorch, label='PyTorch NN', alpha=0.7)
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].set_title('训练损失曲线对比')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 手动NN的预测效果
with torch.no_grad():
    predictions_manual = model_manual.forward(X_tensor)
    predictions_pytorch = model_pytorch(X_tensor)

# 排序以便画线
sort_idx = X_tensor[:, 0].argsort()
X_sorted = X_tensor[sort_idx]
y_sorted = y_tensor[sort_idx]
pred_manual_sorted = predictions_manual[sort_idx]
pred_pytorch_sorted = predictions_pytorch[sort_idx]

# 手动NN预测
axes[0, 1].scatter(X, y, alpha=0.5, label='真实数据')
axes[0, 1].plot(X_sorted.numpy(), pred_manual_sorted.numpy(),
                'r-', linewidth=2, label='手动NN预测')
axes[0, 1].set_xlabel('X')
axes[0, 1].set_ylabel('y')
axes[0, 1].set_title('手动神经网络预测效果')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# PyTorch NN预测
axes[1, 0].scatter(X, y, alpha=0.5, label='真实数据')
axes[1, 0].plot(X_sorted.numpy(), pred_pytorch_sorted.numpy(),
                'b-', linewidth=2, label='PyTorch NN预测')
axes[1, 0].set_xlabel('X')
axes[1, 0].set_ylabel('y')
axes[1, 0].set_title('PyTorch神经网络预测效果')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# 残差分布
residuals_manual = (y_tensor - predictions_manual).numpy()
residuals_pytorch = (y_tensor - predictions_pytorch).numpy()

axes[1, 1].hist(residuals_manual, bins=20, alpha=0.5, label='Manual NN')
axes[1, 1].hist(residuals_pytorch, bins=20, alpha=0.5, label='PyTorch NN')
axes[1, 1].set_xlabel('Residuals')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].set_title('残差分布对比')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(r'E:\python project\ai-learning\learn6\neural_network_comparison.png', dpi=150)
plt.show()

# 6. 模型评估
print("\n6. 模型评估")

def calculate_metrics(y_true, y_pred):
    mse = torch.mean((y_true - y_pred) ** 2).item()
    mae = torch.mean(torch.abs(y_true - y_pred)).item()
    r2 = 1 - torch.sum((y_true - y_pred) ** 2) / torch.sum((y_true - torch.mean(y_true)) ** 2)
    return mse, mae, r2.item()

mse_manual, mae_manual, r2_manual = calculate_metrics(y_tensor, predictions_manual)
mse_pytorch, mae_pytorch, r2_pytorch = calculate_metrics(y_tensor, predictions_pytorch)

print("手动神经网络:")
print(f"  MSE: {mse_manual:.4f}")
print(f"  MAE: {mae_manual:.4f}")
print(f"  R²: {r2_manual:.4f}")

print("\nPyTorch神经网络:")
print(f"  MSE: {mse_pytorch:.4f}")
print(f"  MAE: {mae_pytorch:.4f}")
print(f"  R²: {r2_pytorch:.4f}")

# 保存模型
torch.save(model_pytorch.state_dict(), r'E:\python project\ai-learning\learn6\pytorch_model.pth')
print("\n模型已保存: learn6/pytorch_model.pth")

print("\n 手写神经网络完成")


