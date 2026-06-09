import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data._utils import pin_memory
from torchvision.transforms.functional import to_pil_image

# 设置中文
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("CIFAR-10数据集加载和探索")
print("=" * 60)

# 1. CIFAR-10类别

classes = ['飞机', '汽车', '鸟', '猫', '鹿',
           '狗', '青蛙', '马', '船', '卡车']
classes_en = ['airplane', 'automobile', 'bird', 'cat', 'deer',
              'dog', 'frog', 'horse', 'ship', 'truck']
print(f"类别数量:{len(classes)}")
for i,(c_en,c_zh) in enumerate(zip(classes_en,classes)):
    print(f" {i}: {c_en} ({c_zh})")


# 2. 数据预处理
print("\n2. 数据预处理")

# CIFAR-10的均值和标准差（预计算的）
# 注意：CIFAR-10是彩色图像，3个通道各有自己的均值和标准差
mean = [0.4914, 0.4822, 0.4465]
std = [0.2023, 0.1994, 0.2010]

#训练集：需要数据增强
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),     #随机水平翻转
    transforms.RandomCrop(32,padding=4),   #随机裁剪
    transforms.ToTensor(),
    transforms.Normalize(mean,std)
])

#测试集，只需要归一化
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean,std)
])

print("训练集转换:")
print("  - RandomHorizontalFlip: 随机水平翻转")
print("  - RandomCrop: 随机裁剪(32, padding=4)")
print("  - ToTensor: 转换为张量")
print("  - Normalize: 标准化")

print("\n测试集转换:")
print("  - ToTensor: 转换为张量")
print("  - Normalize: 标准化")

# 3. 下载和加载数据
print("\n3. 下载和加载数据")

train_dataset = torchvision.datasets.CIFAR10(
    root = './data',
    train = True,
    download = True,
    transform = train_transform
)

test_dataset = torchvision.datasets.CIFAR10(
    root = './data',
    train = False,
    download = True,
    transform = test_transform
)

# 同时加载原始图像用于可视化
train_dataset_raw = torchvision.datasets.CIFAR10(
    root = './data',
    train = True,
    download = True,
    transform = transforms.ToTensor()
)

print(f"训练集大小：{len(train_dataset)}")
print(f"测试集大小:{len(test_dataset)}")

# 4. 创建数据加载器
print("\n4. 创建数据加载器")

batch_size = 64

train_loader = torch.utils.data.DataLoader(
    dataset = train_dataset,
    batch_size = batch_size,
    shuffle = True,      #每个epoch随机打乱数据
    num_workers = 0,     #使用几个子进程读取数据
    pin_memory = True    #固定内存（锁页内存）
)
test_load = torch.utils.data.DataLoader(
    dataset = test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers = 0,
    pin_memory=True
)

print(f"批次大小:{batch_size}")
print(f"训练集批次数:{len(train_loader)}")
print(f"测试集批次数:{len(test_load)}")

# 5. 探索数据
print("\n5. 探索数据")

# 获取一个批次
data_iter = iter(train_loader)
images,labels = next(data_iter)

print(f"图像形状: {images.shape}")  # [batch, channels, height, width]
print(f"标签形状: {labels.shape}")
print(f"标签值范围: {labels.min()} - {labels.max()}")
print(f"图像值范围: [{images.min():.3f}, {images.max():.3f}]")

# 6. 可视化原始样本（未归一化）
print("\n6. 可视化原始样本")

fig, axes = plt.subplots(2, 5, figsize=(15, 6))
fig.suptitle('CIFAR-10 原始样本', fontsize=16)

# 获取原始图像
raw_iter = iter(torch.utils.data.DataLoader(train_dataset_raw, batch_size=10, shuffle=True))
raw_images, raw_labels = next(raw_iter)

for i, ax in enumerate(axes.flat):
    # 将张量转换为图像显示
    img = raw_images[i].permute(1, 2, 0).numpy()
    label = raw_labels[i].item()

    ax.imshow(img)
    ax.set_title(f'{classes[label]}', fontsize=10)
    ax.axis('off')

plt.tight_layout()
plt.savefig('./cifar10_samples.png', dpi=150)
plt.show()

# 7. 可视化增强后的样本
print("\n7. 可视化数据增强效果")

# 获取同一个图像的多个增强版本
sample_idx = 0
original_img = train_dataset_raw[sample_idx][0]

fig, axes = plt.subplots(2, 4, figsize=(12, 6))
fig.suptitle('数据增强效果展示', fontsize=14)

for i, ax in enumerate(axes.flat):
    # 应用不同的随机增强
    if i == 0:
        # 原始图像
        img = original_img.permute(1, 2, 0).numpy()
        ax.set_title('原始')
    else:
        # 重新应用训练集转换
        pil_img = to_pil_image(original_img)
        augmented = train_transform(pil_img)
        img = augmented.permute(1, 2, 0).numpy()
        # 反标准化以便显示
        img = img * std + mean
        img = np.clip(img, 0, 1)
        ax.set_title(f'增强版本{i}')

    ax.imshow(img)
    ax.axis('off')

plt.tight_layout()
plt.savefig('./augmentation_demo.png', dpi=150)
plt.show()

# 8. 标签分布
print("\n8. 标签分布")

train_labels = [label for _, label in train_dataset]
unique, counts = np.unique(train_labels, return_counts=True)

plt.figure(figsize=(12, 6))
bars = plt.bar(range(len(classes)), counts)
plt.xlabel('类别')
plt.ylabel('样本数量')
plt.title('CIFAR-10训练集标签分布')
plt.xticks(range(len(classes)), classes, rotation=45)

for bar, count in zip(bars, counts):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
             str(count), ha='center', va='bottom')

plt.savefig('./label_distribution.png', dpi=150)
plt.show()

print("标签分布:")
for i, (c, count) in enumerate(zip(classes, counts)):
    print(f"  {c}: {count} 个样本 ({count/len(train_dataset)*100:.1f}%)")

# 9. 保存数据信息
import json

data_info = {
    'num_classes': 10,
    'classes': classes,
    'classes_en': classes_en,
    'train_size': len(train_dataset),
    'test_size': len(test_dataset),
    'batch_size': batch_size,
    'input_shape': (3, 32, 32),
    'mean': mean,
    'std': std
}

with open('./data_info.json', 'w', encoding='utf-8') as f:
    json.dump(data_info, f, indent=2, ensure_ascii=False)

print("\n数据信息已保存: learn8/data_info.json")

print("\n CIFAR-10数据加载完成")