import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import random

print("=" * 60)
print("模型预测演示")
print("=" * 60)

# 1. 设置设备
print("\n1. 设置设备")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 2. 加载模型
print("\n2. 加载模型")

from cnn_model import CNN

model = CNN().to(device)
model.load_state_dict(torch.load('learn7/best_model.pth', map_location=device))
model.eval()
print("模型加载成功")

#3. 预测函数
def predict_image(model,image_tensor,device):
    """预测单个图像"""
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        # 添加batch维度
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)
        output = model(image_tensor)
        probabilities = torch.softmax(output,dim=1)
        predicted_class = torch.argmax(probabilities,dim=1).item()
        confidence = probabilities[0][predicted_class].item()
    return predicted_class, confidence,probabilities.cpu().numpy()[0]

# 4. 从测试集随机预测
print("\n4. 从测试集随机预测")

# 加载原始测试集
test_dataset = torchvision.datasets.MNIST(
    root='./learn7/data',
    train=False,
    download=False,
    transform=transforms.ToTensor()
)

# 随机选择10个样本
indices = random.sample(range(len(test_dataset)), 10)

fig, axes = plt.subplots(2, 5, figsize=(15, 6))
fig.suptitle('随机样本预测结果', fontsize=16)

for i, (ax, idx) in enumerate(zip(axes.flat, indices)):
    image, true_label = test_dataset[idx]

    # 预测
    pred_class, confidence, probs = predict_image(model, image, device)

    # 显示图像
    ax.imshow(image.squeeze(), cmap='gray')

    # 设置标题颜色（正确绿色，错误红色）
    color = 'green' if pred_class == true_label else 'red'
    ax.set_title(f'真实: {true_label}\n预测: {pred_class} ({confidence:.2%})',
                 color=color, fontsize=10)
    ax.axis('off')

plt.tight_layout()
plt.savefig('./predictions_random.png', dpi=150)
plt.show()

# 5. 概率分布可视化
print("\n4. 概率分布可视化")

# 选择一个样本
idx = random.randint(0, len(test_dataset))
image, true_label = test_dataset[idx]
pred_class, confidence, probs = predict_image(model, image, device)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 显示图像
axes[0].imshow(image.squeeze(), cmap='gray')
axes[0].set_title(f'真实标签: {true_label}\n预测标签: {pred_class}\n置信度: {confidence:.2%}')
axes[0].axis('off')

# 显示概率分布
classes = range(10)
colors = ['red' if i == pred_class else 'blue' for i in classes]
axes[1].bar(classes, probs, color=colors)
axes[1].set_xlabel('数字类别')
axes[1].set_ylabel('概率')
axes[1].set_title('各类别预测概率')
axes[1].set_xticks(classes)

plt.tight_layout()
plt.savefig('./probability_distribution.png', dpi=150)
plt.show()

# 6. 批量预测统计
print("\n5. 批量预测统计")

# 加载归一化的测试集
transform_norm = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

test_dataset_norm = torchvision.datasets.MNIST(
    root='./learn7/data',
    train=False,
    download=False,
    transform=transform_norm
)

test_loader = torch.utils.data.DataLoader(test_dataset_norm, batch_size=64, shuffle=False)

all_confidences = []
all_correct = []

with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        probabilities = torch.softmax(output, dim=1)
        confidences, predictions = probabilities.max(dim=1)

        all_confidences.extend(confidences.cpu().numpy())
        all_correct.extend((predictions == target).cpu().numpy())

print(f"平均置信度: {np.mean(all_confidences):.4f}")
print(f"正确预测的平均置信度: {np.mean([c for i, c in enumerate(all_confidences) if all_correct[i]]):.4f}")
print(f"错误预测的平均置信度: {np.mean([c for i, c in enumerate(all_confidences) if not all_correct[i]]):.4f}")

# 7. 置信度分布
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(all_confidences, bins=50, alpha=0.7, edgecolor='black')
plt.xlabel('置信度')
plt.ylabel('样本数')
plt.title('预测置信度分布')

plt.subplot(1, 2, 2)
correct_confs = [c for i, c in enumerate(all_confidences) if all_correct[i]]
wrong_confs = [c for i, c in enumerate(all_confidences) if not all_correct[i]]
plt.hist(correct_confs, bins=50, alpha=0.7, label='正确预测', color='green')
plt.hist(wrong_confs, bins=50, alpha=0.7, label='错误预测', color='red')
plt.xlabel('置信度')
plt.ylabel('样本数')
plt.title('正确vs错误预测的置信度分布')
plt.legend()

plt.tight_layout()
plt.savefig('./confidence_distribution.png', dpi=150)
plt.show()

# 8. 保存模型为可部署格式
print("\n6. 保存模型为可部署格式")

# 保存完整模型
torch.save(model, './full_model.pth')

# 导出为TorchScript（用于C++部署）
example_input = torch.randn(1, 1, 28, 28).to(device)
traced_model = torch.jit.trace(model, example_input)
traced_model.save('./model_traced.pt')

print("模型已保存:")
print("  - learn7/full_model.pth (完整模型)")
print("  - learn7/model_traced.pt (TorchScript格式)")

# 9. 创建简单的预测API
print("\n7. 创建简单预测函数")


def predict_from_array(image_array):
    """
    从numpy数组预测
    image_array: shape (28, 28), dtype uint8, 值范围0-255
    """
    # 预处理
    image_tensor = torch.FloatTensor(image_array).unsqueeze(0).unsqueeze(0) / 255.0
    transform = transforms.Normalize((0.1307,), (0.3081,))
    image_tensor = transform(image_tensor)

    pred_class, confidence, probs = predict_image(model, image_tensor, device)
    return pred_class, confidence


print("预测函数已创建: predict_from_array(image_array)")
print("\n示例:")
# 创建简单测试
test_array = np.random.randint(0, 255, (28, 28))
pred, conf = predict_from_array(test_array)
print(f"  随机数组预测结果: {pred} (置信度: {conf:.2%})")

print("\n 模型预测演示完成")