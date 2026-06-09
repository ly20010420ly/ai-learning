import torch
import torch.nn as nn
import torch.nn.functional as F

print("=" * 60)
print("CIFAR-10 CNN模型构建")
print("=" * 60)

# 1. 基础CNN模型
print("\n1. 基础CNN模型")

class BasicCNN(nn.Module):
    """
    基础CNN模型用于CIFAR-10
    架构: Conv1 -> Conv2 -> Conv3 -> FC1 -> FC2
    """
    def __init__(self,num_classes=10):
        super(BasicCNN, self).__init__()
        #卷积层
        self.conv1 = nn.Conv2d(3,32,kernel_size=3,padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32,64,kernel_size=3,padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64,128,kernel_size=3,padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        #池化层
        self.pool = nn.MaxPool2d(2,2)

        #全连接层
        self.fc1 = nn.Linear(128*4*4,512)
        self.fc2 = nn.Linear(512,num_classes)

        #Dropout
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        #第一个卷积快
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        #第二个卷积快
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        #第三个卷积快
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        #展平
        x = x.view(x.size(0), -1)
        #全连接层
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# 2. 改进版CNN（使用残差连接）
print("\n2. 改进版CNN（残差块）")

class ResidualBlock(nn.Module):
    """残差快"""
    def __init__(self,input_channel,output_channel,stride=1):
        super(ResidualBlock,self).__init__()

        self.conv1 = nn.Conv2d(input_channel,output_channel,
                               kernel_size=3,stride=stride,padding=1)
        self.bn1 = nn.BatchNorm2d(output_channel)

        self.conv2 = nn.Conv2d(output_channel,output_channel,
                               kernel_size=3,stride=1,padding=1)
        self.bn2 = nn.BatchNorm2d(output_channel)

        #跳跃连接
        self.shortcut = nn.Sequential()
        if stride != 1 or input_channel != output_channel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(input_channel,output_channel,
                          kernel_size=1,stride=stride),
                nn.BatchNorm2d(output_channel)
            )
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNetCNN(nn.Module):
    """带残差连接的CNN"""
    def __init__(self,num_classes=10):
        super(ResNetCNN, self).__init__()

        self.conv1 = nn.Conv2d(3,32,kernel_size=3,padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.layer1 = ResidualBlock(32,32)
        self.layer2 = ResidualBlock(32,64,stride=2)
        self.layer3 = ResidualBlock(64,128,stride=2)

        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(128,num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# 3. 创建模型实例
print("\n3. 创建模型实例")

basic_model = BasicCNN()
resnet_model = ResNetCNN()

print(f"基础CNN参数数量:{sum(p.numel() for p in basic_model.parameters()):,}")
print(f"RestNet CNN参数数量:{sum(p.numel() for p in resnet_model.parameters()):,}")

# 4. 测试前向传播
print("\n4. 测试前向传播")

batch_size = 64
test_input = torch.randn(batch_size,3,32,32)

with torch.no_grad():
    basic_output = basic_model(test_input)
    resnet_output = resnet_model(test_input)

print(f"输入形状: {test_input.shape}")
print(f"基础CNN输出形状: {basic_output.shape}")
print(f"ResNet输出形状: {resnet_output.shape}")

# 5. 可视化模型结构
print("\n5. 可视化模型结构")

# 打印基础CNN的结构
print("\n基础CNN结构:")
print(basic_model)

print("\nResNet CNN结构:")
print(resnet_model)

# 6. 保存模型定义
import joblib

joblib.dump(basic_model, './basic_cnn_definition.pkl')
joblib.dump(resnet_model, './resnet_cnn_definition.pkl')

print("\n模型定义已保存")
print("  - learn8/basic_cnn_definition.pkl")
print("  - learn8/resnet_cnn_definition.pkl")

print("\n CIFAR-10 CNN模型构建完成")



