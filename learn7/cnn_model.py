import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt

# 设置中文
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("构建CNN模型")
print("=" * 60)

# 1. 定义CNN架构
print("\n1. 定义CNN架构")
class CNN(nn.Module):
    """
    CNN模型用于MNIST分类
    架构：Conv1 -> Pool1 -> Conv2 -> Pool2 -> FC1 -> FC2
    """
    def __init__(self):
        super(CNN,self).__init__()

        #第一个卷积块：输入通道1 -> 输出通道32
        self.conv1 = nn.Conv2d(
            in_channels=1,    #输入通道（灰度图）
            out_channels=32,  #输出通道（卷积核数量）
            kernel_size=3,    #卷积核大小
            stride=1,         #步长
            padding=1         #填充，保持尺寸
        )
        self.bn1 = nn.BatchNorm2d(32) #批归一化

        #第二个卷积核：32通道 -> 64通道
        self.conv2 = nn.Conv2d(32,64,kernel_size=3,stride=1,padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        #池化层
        self.pool = nn.MaxPool2d(kernel_size=2,stride=2)

        #全连接层
        #经过两次池化后： 28*28 -> 14*14 -> 7*7
        #输入：64 *7 *7 = 3136
        self.fc1 = nn.Linear(64*7*7,128)
        self.fc2 = nn.Linear(128,10)

        #Dorpout防止过拟合
        self.dropout = nn.Dropout(0.25)

    def forward(self,x):
        #第一个卷积核
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)
        #第二个卷积核
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)
        #展平
        x = x.view(x.size(0),-1)
        #全连接层
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.dropout(x)

        return x


# 2. 创建一个简单版本的CNN（用于理解）
print("\n2. 简单CNN版本（用于理解）")

class SimpleCNN(nn.Module):
    """
    更简单的CNN构架，便于理解
    """
    def __init__(self):
        super(SimpleCNN,self).__init__()
        self.conv1 = nn.Conv2d(1,16,3,padding=1)
        self.conv2 = nn.Conv2d(16,32,3,padding=1)
        self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(32*7*7,128)
        self.fc2 = nn.Linear(128,10)

    def forward(self,x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        x = x.view(x.size(0),-1)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# 3. 创建模型实例
print("\n3. 创建模型实例")

model = CNN()
simple_model = SimpleCNN()

print(f"CNN模型结构：\n{model}")
print(f"\n模型参数数量：{sum(p.numel() for p in model.parameters()):,}")


# 4. 测试前向传播
print("\n4. 测试前向传播")

# 创建模拟输入
batch_size = 4
test_input = torch.randn(batch_size,1,28,28)

print(f"输入形状:{test_input.shape}")

#前向传播
with torch.no_grad():
    output = model(test_input)
    simple_output = simple_model(test_input)

print(f"输出形状:{output.shape}")
print(f"输出值范围:[{output.min():.3f}, {output.max():.3f}]")

# 5. 查看各层输出形状
print("\n5. 查看各层输出形状")


class ShapePrinter(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        print(f"输入: {x.shape}")
        x = self.model.conv1(x)
        print(f"conv1后: {x.shape}")
        x = self.model.bn1(x)
        print(f"bn1后: {x.shape}")
        x = F.relu(x)
        print(f"relu后: {x.shape}")
        x = self.model.pool(x)
        print(f"pool后: {x.shape}")
        x = self.model.conv2(x)
        print(f"conv2后: {x.shape}")
        x = self.model.bn2(x)
        print(f"bn2后: {x.shape}")
        x = F.relu(x)
        print(f"relu后: {x.shape}")
        x = self.model.pool(x)
        print(f"pool后: {x.shape}")
        x = x.view(x.size(0), -1)
        print(f"展平后: {x.shape}")
        x = self.model.fc1(x)
        print(f"fc1后: {x.shape}")
        x = self.model.dropout(x)
        print(f"dropout后: {x.shape}")
        x = self.model.fc2(x)
        print(f"fc2后: {x.shape}")
        return x


shape_printer = ShapePrinter(model)
with torch.no_grad():
    shape_printer(test_input)


# 6. 可视化卷积核
print("\n6. 可视化卷积核")

fig, axes = plt.subplots(4, 8, figsize=(12, 6))
fig.suptitle('第1层卷积核可视化', fontsize=14)

# 获取第一层卷积核
conv1_weights = model.conv1.weight.data

for i, ax in enumerate(axes.flat):
    if i < conv1_weights.shape[0]:
        kernel = conv1_weights[i, 0].numpy()
        ax.imshow(kernel, cmap='coolwarm')
        ax.set_title(f'核{i+1}', fontsize=8)
    ax.axis('off')

plt.tight_layout()
plt.savefig(r'E:\python project\ai-learning\learn7\conv1_filters.png', dpi=150)
plt.show()

# 7. 保存模型定义
import joblib
joblib.dump(model, r'E:\python project\ai-learning\learn7\model_definition.pkl')
print("\n模型定义已保存: learn7/model_definition.pkl")

print("\n CNN模型构建完成")
