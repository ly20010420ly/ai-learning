import torch
import torchvision.models as models
import torch.nn as nn
import matplotlib.pyplot as plt

# 设置中文
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


print("=" * 60)
print("迁移学习理论基础")
print("=" * 60)

# 1. 什么是迁移学习
print("\n1. 迁移学习简介")

print("""
迁移学习：将在一个任务上学习到的知识应用到另一个相关任务上。

优势：
  - 减少训练时间（无需从头训练）
  - 需要更少的数据（预训练模型已有通用特征）
  - 更好的性能（特别是在小数据集上）

迁移学习策略：
  1. 特征提取：冻结预训练模型，只训练新分类头
  2. 微调：解冻部分层，用新数据继续训练
  3. 两者结合：先特征提取，再微调
""")


# 2. 预训练模型对比
print("\n2. 预训练模型对比")

# 列出可用的预训练模型
print("torchvision提供的预训练模型:")
models_list = [
    ('ResNet18', '11.7M参数, ImageNet Top-1: 69.8%'),
    ('ResNet34', '21.8M参数, ImageNet Top-1: 73.3%'),
    ('ResNet50', '25.6M参数, ImageNet Top-1: 76.1%'),
    ('ResNet101', '44.5M参数, ImageNet Top-1: 77.4%'),
    ('ResNet152', '60.2M参数, ImageNet Top-1: 78.3%'),
    ('VGG16', '138M参数, ImageNet Top-1: 71.6%'),
    ('DenseNet121', '8.0M参数, ImageNet Top-1: 74.4%'),
    ('MobileNetV2', '3.5M参数, ImageNet Top-1: 71.9%'),
    ('EfficientNet_B0', '5.3M参数, ImageNet Top-1: 77.7%')
]

for name,info in models_list:
    print(f" {name:15s}:{info}")

# 3. ResNet架构详解
print("\n3. ResNet架构详解")


class ResidualBlock:
    """残差块原理"""

    def __init__(self):
        pass

    def explain(self):
        print("""
ResNet的核心创新：残差连接（Skip Connection）

传统CNN: x -> Conv -> ReLU -> Conv -> ReLU -> H(x)
ResNet:   x -> Conv -> ReLU -> Conv -> ReLU -> H(x) + x -> F(x) + x

优势：
  - 解决梯度消失问题
  - 可以训练更深的网络
  - 恒等映射保证性能不退化
        """)


ResidualBlock().explain()

# 4. 加载预训练模型
print("\n4. 加载预训练ResNet18")

#加载预训练的RestNet18
resnet18 = models.resnet18(pretrained=True)
print('Resnet18加载成功')
print(f"模型类型：{type(resnet18)}")

#查看模型结构
print("模型结构")
print(f"  -卷积层：{len([m for m in resnet18.modules() if isinstance(m,nn.Conv2d)])}个")
print(f"  -全连接层：{len([m for m in resnet18.modules() if isinstance(m,nn.Linear)])}个")
print(f"  -BN层：{len([m for m in resnet18.modules() if isinstance(m,nn.BatchNorm2d)])}个")
print(f"  -总参数量：{sum(p.numel() for p in resnet18.parameters()):,}个")

# 5. 查看网络结构
print("\n5. ResNet18结构分析")

#打印前几层
print("ResNet18前几层结构")
for name,module in list(resnet18.named_children())[:5]:
    print(f" {name}:{module}")

print("\n分类头：")
print(f"   fc:{resnet18.fc}")

# 6. 特征维度分析
print("\n6. 特征维度分析")

#  模拟前向传播获取特征维度
dummy_input = torch.randn(1,3,224,224)

# 特征提取器（去掉最后的fc层）
feature_extractor = nn.Sequential(*list(resnet18.children())[:-1])
with torch.no_grad():
    features = feature_extractor(dummy_input)

print(f"输入形状：{dummy_input.shape}")
print(f"输出形状：{features.shape}")
print(f"特征向量维度：{features.numel()}")

# 7. 可视化ResNet结构
print("\n7. 可视化ResNet层次")

def count_layers(model):
    """统计各类型层数"""
    counts = {
        'Conv2d': 0,
        'BatchNorm2d': 0,
        'ReLU': 0,
        'MaxPool2d': 0,
        'AvgPool2d': 0,
        'Linear': 0
    }

    for module in model.modules():
        for layer_type in counts:
            if isinstance(module,getattr(nn,layer_type)):
                counts[layer_type] += 1

    return counts

layer_counts = count_layers(resnet18)

plt.figure(figsize=(10, 6))
layers = list(layer_counts.keys())
counts = list(layer_counts.values())
bars = plt.bar(layers, counts)
plt.xlabel('层类型')
plt.ylabel('数量')
plt.title('ResNet18各类型层统计')
for bar, count in zip(bars, counts):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             str(count), ha='center', va='bottom')
plt.savefig('./resnet_layers.png', dpi=150)
plt.show()

print(f"\n层统计:")
for layer_type, count in layer_counts.items():
    print(f"  {layer_type}: {count}")

# 8. 保存模型信息
import json

model_info = {
    'name': 'ResNet18',
    'pretrained': True,
    'num_parameters': sum(p.numel() for p in resnet18.parameters()),
    'input_size': (3, 224, 224),
    'output_features': 512,
    'layer_counts': layer_counts
}

with open('./model_info.json', 'w') as f:
    json.dump(model_info, f, indent=2)

print("\n 迁移学习理论完成")







