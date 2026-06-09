import torch
import torch.nn as nn
import torch.optim as optim
from torch.ao.pruning import scheduler
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import time
import os

from learn6.optimizers_losses import current_lr

print("=" * 60)
print("CNN模型训练")
print("=" * 60)

# 1. 设置设备
print("\n1. 设置训练设备")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备:{device}")
if torch.cuda.is_available():
    print(f"GPU型号：{torch.cuda.get_device_name(0)}")

#加载数据
print("\n2. 加载数据")

transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])

train_dataset = torchvision.datasets.MNIST(
    root="./learn7/data",
    train=True,
    download=False,
    transform=transform,
)

test_dataset = torchvision.datasets.MNIST(
    root="./learn7/data",
    train=False,
    download=False,
    transform=transform,
)

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,num_workers=0)

print(f"训练集：{len(train_dataset)} 样本，{len(train_loader)} 批次")
print(f"测试集：{len(test_dataset)} 样本，{len(test_loader)} 批次")

# 3. 创建模型
print("\n3. 创建模型")

from cnn_model import CNN

model = CNN().to(device)
print(f"模型参数数量：{sum(p.numel() for p in model.parameters()):,}")


# 4. 定义损失函数和优化器
print("\n4. 定义损失函数和优化器")

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
#学习率调度器
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

print(f"损失函数: CrossEntropyLoss")
print(f"优化器: Adam (lr=0.001)")
print(f"学习率调度: StepLR (step=5, gamma=0.5)")


# 5. 训练函数
print("\n5. 开始训练")

def train_epoch(model,train_loader,criterion,optimizer,device):
    """训练一个epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx,(data,target) in enumerate(tqdm(train_loader,desc='训练')):
        data, target = data.to(device), target.to(device)
        # 清零梯度
        optimizer.zero_grad()
        #前向传播
        output = model(data)
        loss = criterion(output,target)
        #反向传播
        loss.backward()
        optimizer.step()
        #统计
        running_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total

    return epoch_loss, epoch_acc

def validate(model,test_loader,criterion,device):
    """验证模型"""
    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for data,target in tqdm(test_loader,desc='验证'):
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output,target).item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

    test_loss = test_loss / len(test_loader)
    test_acc = 100. * correct / total

    return test_loss, test_acc


# 6. 训练循环
print("\n6. 训练循环")

num_epochs = 10
train_losses = []
train_accs = []
test_losses = []
test_accs = []
best_acc = 0

start_time = time.time()

for epoch in range(1,num_epochs+1):
    print(f"\nEpoch {epoch}/{num_epochs}")
    print("-" * 40)

    #训练
    train_loss,train_acc = train_epoch(model,train_loader,criterion,optimizer,device)
    train_losses.append(train_loss)
    train_accs.append(train_acc)

    #验证
    test_loss,test_acc = validate(model,test_loader,criterion,device)
    test_losses.append(test_loss)
    test_accs.append(test_acc)

    # 更新学习率
    scheduler.step()
    current_lr = optimizer.param_groups[0]['lr']

    #打印结果
    print(f"训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.2f}%")
    print(f"验证损失: {test_loss:.4f}, 验证准确率: {test_acc:.2f}%")
    print(f"学习率: {current_lr:.6f}")

    #保存最佳模型
    if test_acc > best_acc:
        best_acc = test_acc
        torch.save(model.state_dict(),'learn7/best_model.pth')
        print(f" 保存最佳模型 (准确率: {best_acc:.2f}%)")

training_time = time.time() - start_time
print(f"\n训练完成！总用时: {training_time:.2f} 秒")
print(f"最佳验证准确率: {best_acc:.2f}%")

# 7. 可视化训练过程
print("\n7. 可视化训练过程")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 损失曲线
axes[0, 0].plot(train_losses, label='训练损失', linewidth=2)
axes[0, 0].plot(test_losses, label='验证损失', linewidth=2)
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].set_title('损失曲线')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 准确率曲线
axes[0, 1].plot(train_accs, label='训练准确率', linewidth=2)
axes[0, 1].plot(test_accs, label='验证准确率', linewidth=2)
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Accuracy (%)')
axes[0, 1].set_title('准确率曲线')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 最终准确率对比
epochs_range = range(1, num_epochs + 1)
axes[1, 0].bar(epochs_range, train_accs, alpha=0.7, label='训练')
axes[1, 0].bar(epochs_range, test_accs, alpha=0.7, label='验证')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Accuracy (%)')
axes[1, 0].set_title('训练 vs 验证准确率')
axes[1, 0].legend()

# 学习率变化
axes[1, 1].plot(epochs_range, [optimizer.param_groups[0]['lr']] * num_epochs,
                marker='o', linewidth=2)
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Learning Rate')
axes[1, 1].set_title('学习率变化')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('./training_history.png', dpi=150)
plt.show()

# 8. 保存训练记录
import joblib

training_info = {
    'train_losses': train_losses,
    'train_accs': train_accs,
    'test_losses': test_losses,
    'test_accs': test_accs,
    'best_acc': best_acc,
    'training_time': training_time,
    'num_epochs': num_epochs
}

joblib.dump(training_info, './training_info.pkl')
print("\n训练记录已保存: learn7/training_info.pkl")

print("\n 模型训练完成")































