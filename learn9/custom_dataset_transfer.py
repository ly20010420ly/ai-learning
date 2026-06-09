import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os

print("=" * 60)
print("迁移学习在自定义数据集上的应用")
print("=" * 60)

# 设置中文
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 1. 创建模拟的自定义数据集
print("\n1. 创建自定义数据集")


class CustomDataset(Dataset):
    """自定义数据集示例"""

    def __init__(self, num_samples=500, num_classes=5, img_size=224, transform=None):
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.img_size = img_size
        self.transform = transform

        # 生成随机图像和标签
        self.images = np.random.randint(0, 255,
                                        (num_samples, img_size, img_size, 3),
                                        dtype=np.uint8)
        self.labels = np.random.randint(0, num_classes, num_samples)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]

        # 转换为PIL图像
        img = Image.fromarray(img)

        if self.transform:
            img = self.transform(img)

        # 关键修复：将标签转换为Long类型
        return img, torch.tensor(label, dtype=torch.long)


# 自定义类别
custom_classes = ['类A', '类B', '类C', '类D', '类E']
print(f"自定义类别: {custom_classes}")

# 2. 数据预处理
print("\n2. 数据预处理")

# 使用ImageNet的均值和标准差（因为使用预训练模型）
mean_imagenet = [0.485, 0.456, 0.406]
std_imagenet = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean_imagenet, std_imagenet)
])

test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean_imagenet, std_imagenet)
])

# 创建数据集
train_dataset = CustomDataset(num_samples=400, transform=train_transform)
val_dataset = CustomDataset(num_samples=100, transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

print(f"训练集大小: {len(train_dataset)}")
print(f"验证集大小: {len(val_dataset)}")

# 3. 构建迁移学习模型
print("\n3. 构建迁移学习模型")


class TransferModel(nn.Module):
    def __init__(self, num_classes=5):
        super(TransferModel, self).__init__()
        # 使用预训练的ResNet18
        self.backbone = torchvision.models.resnet18(pretrained=True)

        # 冻结所有层
        for param in self.backbone.parameters():
            param.requires_grad = False

        # 替换分类头
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)


model = TransferModel(num_classes=5)
print(f"模型总参数: {sum(p.numel() for p in model.parameters()):,}")
print(f"可训练参数: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# 4. 训练模型
print("\n4. 训练模型")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 20
train_losses = []
val_accs = []

for epoch in range(1, epochs + 1):
    # 训练
    model.train()
    train_loss = 0
    correct = 0
    total = 0

    for data, target in train_loader:
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        # 计算训练准确率
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

    train_loss /= len(train_loader)
    train_acc = 100. * correct / total
    train_losses.append(train_loss)

    # 验证
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

    val_acc = 100. * correct / total
    val_accs.append(val_acc)

    if epoch % 5 == 0:
        print(f"Epoch {epoch}: Loss={train_loss:.4f}, Train Acc={train_acc:.2f}%, Val Acc={val_acc:.2f}%")

# 5. 可视化结果
print("\n5. 可视化训练结果")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(train_losses)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('训练损失')
axes[0].grid(True, alpha=0.3)

axes[1].plot(val_accs)
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy (%)')
axes[1].set_title('验证准确率')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('./custom_dataset_results.png', dpi=150)
plt.show()

print(f"\n最终验证准确率: {val_accs[-1]:.2f}%")

# 6. 迁移学习最佳实践总结
print("\n6. 迁移学习最佳实践")

best_practices = """
迁移学习最佳实践:

1. 数据准备:
   - 使用与预训练模型相同的预处理
   - ImageNet模型使用[0,1]归一化
   - 调整图像大小到224x224

2. 模型选择:
   - 小数据集: ResNet18, MobileNet
   - 大数据集: ResNet50, EfficientNet
   - 考虑推理速度: MobileNet, EfficientNet-lite

3. 训练策略:
   - 先训练分类头几个epoch
   - 再解冻最后几层进行微调
   - 使用较低的学习率

4. 调试建议:
   - 如果准确率低，检查数据预处理
   - 如果过拟合，增加Dropout或数据增强
   - 如果欠拟合，解冻更多层

5. 常见应用场景:
   - 医疗图像分类
   - 工业缺陷检测
   - 野生动物识别
   - 自定义物体识别
"""

print(best_practices)

# 保存模型
os.makedirs('./learn9', exist_ok=True)
torch.save(model.state_dict(), '../data and pt/learn9/custom_dataset_model.pth')
print("\n模型已保存: learn9/custom_dataset_model.pth")

print("\n 自定义数据集迁移学习完成")