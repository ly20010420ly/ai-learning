import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np


# 设置中文
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


print("=" * 60)
print("MNIST数据集加载和探索")
print("=" * 60)

# 1. 数据预处理
print("\n1. 数据预处理")

#定义数据转换
# ToTensor():将PIL图像转换为张量，并归一化到[0,1]
# Normalize():标准化：(x - mean) / std
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,),(0.3081,))
])

print("数据转换:")
print("  - ToTensor: 将图像转换为张量，值范围[0,1]")
print("  - Normalize: 标准化，均值0.1307，标准差0.3081")


# 2. 下载和加载训练集
print("\n2. 加载训练集")

train_dataset = torchvision.datasets.MNIST(
    root = './learn7/data',
    train = True,
    download = True,
    transform = transform
)

test_dataset = torchvision.datasets.MNIST(
    root = './learn7/data',
    train = False,
    download = True,
    transform = transforms
)

print(f"训练集大小: {len(train_dataset)}")
print(f"测试集大小: {len(test_dataset)}")

# 3. 创建数据加载器
print("\n3. 创建数据加载器")

batch_size = 64

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size = batch_size,
    shuffle = True,      #训练集打乱顺序
    num_workers = 0      #Windows下建议设为0
)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size = batch_size,
    shuffle = False,
    num_workers = 0
)

print(f"批次大小: {batch_size}")
print(f"训练集批次数: {len(train_loader)}")
print(f"测试集批次数: {len(test_loader)}")


# 4. 探索数据
print("\n4. 探索数据")

# 获取一个批次的数据
data_iter = iter(train_loader)
images, labels = next(data_iter)

print(f"图像张量形状:{images.shape}")   #[batch,channels,height,width]
print(f"标签形状:{labels.shape}")
print(f"图像值范围:{images.min():.3f},{images.max():.3f}")
print(f"标签值范围:{labels.min()},{labels.max()}")


# 5. 可视化样本
print("\n5. 可视化训练样本")

fig,axes = plt.subplots(2,5,figsize=(12,6))
fig.suptitle("MINST手写数字样本",fontsize=16)

for i,ax in enumerate(axes.flat):
    # 获取图像（需要反标准化以便显示）
    img = images[i][0].numpy()
    label = labels[i].item()

    ax.imshow(img,cmap="gray")
    ax.set_title(f"数字：{label}")
    ax.axis('off')

plt.tight_layout()
plt.savefig(r'E:\python project\ai-learning\learn7\mnist_samples.png', dpi=150)
plt.show()


# 6. 标签分布
print("\n6. 标签分布")

#统计训练集标签分布
train_labels = [label for _, label in train_dataset]
unique,counts = np.unique(train_labels,return_counts=True)

plt.figure(figsize=(10,6))
plt.bar(unique,counts)
plt.xlabel('数字')
plt.ylabel('样本数量')
plt.title("MNIST训练集标签分布")
plt.xticks(unique)
for i,(u,c) in enumerate(zip(unique,counts)):
    plt.text(u,c+50,str(c),ha='center')
plt.savefig(r'E:\python project\ai-learning\learn7\label_distribution.png', dpi=150)
plt.show()

print("标签分布:")
for u, c in zip(unique, counts):
    print(f"  数字 {u}: {c} 个样本 ({c/len(train_dataset)*100:.1f}%)")


# 7. 保存数据加载器信息
import joblib

data_info = {
    'train_size':len(train_dataset),
    'test_size':len(test_dataset),
    'batch_size':batch_size,
    'num_classes':10,
    'input_shape':(1,28,28),
    'mean':0.1307,
    'std':0.3081,
}

joblib.dump(data_info, r'E:\python project\ai-learning\learn7\data_info.pkl')
print("\n数据信息已保存: learn7/data_info.pkl")

print("\n MNIST数据加载完成")