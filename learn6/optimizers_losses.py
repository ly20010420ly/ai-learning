import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# 设置中文
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("优化器和损失函数深入")
print("=" * 60)

# 1. 生成数据集
print("\n1. 生成数据集")

np.random.seed(42)
torch.manual_seed(42)

#创建一个简单的分类数据集
n_samples = 500
X = torch.randn(n_samples,2)
y = (X[:, 0] ** 2 + X[:, 1] ** 2 > 1).float().unsqueeze(1)

print(f"x形状:{X.shape}")
print(f"y形状:{y.shape}")
print(f"正样本比例:{y.mean().item():.3f}")

# 2. 定义模型
class SimpleClassifier(nn.Module):
    def __init__(self):
        super(SimpleClassifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2,10),
            nn.Linear(10,10),
            nn.ReLU(),
            nn.Linear(10,1),
            nn.Sigmoid()
        )
    def forward(self,x):
        return self.net(x)

# 3. 比较不同的优化器
print("\n3. 比较不同的优化器")

optimizers_dict = {
    'SGD': optim.SGD,
    'Momentum':lambda params,lr:optim.SGD(params,lr,momentum=0.9),
    'Adam': optim.Adam,
    'RMSprop': optim.RMSprop
}

learning_rates = {
    'SGD':0.1,
    'Momentum':0.1,
    'Adam':0.01,
    'RMSprop':0.01
}

criterion = nn.BCELoss()
epochs = 200
losses_dict = {}

for name,opt_class in optimizers_dict.items():
    print(f"\n训练 {name}...")
    #初始化模型
    model = SimpleClassifier()
    #创建优化器
    if name == 'Momentum':
        optimizer = opt_class(model.parameters(), lr=learning_rates[name])
    else:
        optimizer = opt_class(model.parameters(), lr=learning_rates[name])

    #记录损失
    losses = []

    for epoch in range(epochs):
        #前向传播
        output = model(X)
        loss = criterion(output,y)
        losses.append(loss.item())
        #反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch + 1}, Loss: {loss.item():.4f}")

    losses_dict[name] = losses


# 4. 可视化优化器对比
print("\n4. 可视化优化器对比")
plt.figure(figsize=(12, 6))

for name,losses in losses_dict.items():
    plt.plot(losses,label=name,linewidth=2)

plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title('不同优化器的训练损失对比')
plt.legend()
plt.grid(True,alpha=0.3)
plt.savefig(r'E:\python project\ai-learning\learn6\optimizers_comparison.png', dpi=150)
plt.show()

# 5. 比较不同的损失函数
print("\n5. 比较不同的损失函数")

#生成回归数据
X_reg = torch.randn(200,1)
y_reg = 2 * X_reg + 1 + torch.randn(200, 1) * 0.3

class SimpleRegressor(nn.Module):
    def __init__(self):
        super(SimpleRegressor, self).__init__()
        self.fc1 = nn.Linear(1,10)
        self.fc2 = nn.Linear(10,1)

    def forward(self,x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

loss_functions = {
    'MSE': nn.MSELoss(),
    'MAE': nn.L1Loss(),
    'SmoothL1': nn.SmoothL1Loss()
}

losses_loss_functions = {}

for name,loss_fn in loss_functions.items():
    print(f"\n使用 {name} 损失函数...")

    model = SimpleRegressor()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    losses = []
    for epoch in range(300):
        prediction = model(X_reg)
        loss = loss_fn(prediction,y_reg)
        losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f"  Epoch {epoch + 1}, Loss: {loss.item():.4f}")

    losses_loss_functions[name] = losses

# 可视化损失函数对比
plt.figure(figsize=(12, 6))

for name,losses in losses_loss_functions.items():
    plt.plot(losses,label=name,linewidth=2)

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('不同损失函数的训练对比')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(r'E:\python project\ai-learning\learn6\loss_functions_comparison.png', dpi=150)
plt.show()


# 6. 学习率的影响
print("\n6. 学习率的影响")

learning_rates_test = [0.001,0.01,0.1,1.0]
lr_results = {}

for lr in learning_rates_test:
    print(f"\n学习率 = {lr}")

    model = SimpleRegressor()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    losses = []
    for epoch in range(200):
        predictions = model(X_reg)
        loss = criterion(predictions,y_reg)
        losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if loss.item() > 1e6:  # 梯度爆炸
            print(f"  学习率 {lr} 导致梯度爆炸!")
            losses = [float('inf')] * 200
            break

    lr_results[lr] = losses
    print(f"  最终损失: {losses[-1]:.4f}")

# 可视化学习率影响
plt.figure(figsize=(12, 6))

for lr, losses in lr_results.items():
    if len(losses) == 200:
        plt.plot(losses, label=f'LR = {lr}', linewidth=2)

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('学习率对训练的影响')
plt.legend()
plt.grid(True, alpha=0.3)
plt.yscale('log')  # 对数刻度
plt.savefig(r'E:\python project\ai-learning\learn6\learning_rate_effect.png', dpi=150)
plt.show()


# 7. 学习率调度器
print("\n7. 学习率调度器")

model = SimpleClassifier()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 创建学习率调度器
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

lr_history = []
loss_history = []

for epoch in range(200):
    outputs = model(X)
    loss = criterion(outputs, y)
    loss_history.append(loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 记录学习率
    current_lr = optimizer.param_groups[0]['lr']
    lr_history.append(current_lr)

    # 更新学习率
    scheduler.step()

# 可视化学习率变化
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].plot(loss_history)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('使用学习率调度器的训练损失')
axes[0].grid(True, alpha=0.3)

axes[1].plot(lr_history, 'r-', linewidth=2)
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Learning Rate')
axes[1].set_title('学习率衰减过程')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(r'E:\python project\ai-learning\learn6\learning_rate_scheduler.png', dpi=150)
plt.show()

print("\n 优化器和损失函数对比完成")