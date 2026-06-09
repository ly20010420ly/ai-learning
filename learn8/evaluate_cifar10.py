import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# 设置中文
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("CIFAR-10 模型评估")
print("=" * 60)

# 1. 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备:{device}")

# 2. 加载模型
print("\n1. 加载模型")

from cifar10_cnn import ResNetCNN

model = ResNetCNN().to(device)
model.load_state_dict(torch.load('./best_ResNetCNN.pth', map_location=device))
model.eval()

# 3. 加载测试数据
print("\n2. 加载测试数据")

classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']
classes_zh = ['飞机', '汽车', '鸟', '猫', '鹿',
              '狗', '青蛙', '马', '船', '卡车']

mean = [0.4914, 0.4822, 0.4465]
std = [0.2023, 0.1994, 0.2010]

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean,std)
])

test_dataset = torchvision.datasets.CIFAR10(
    root='./data',
    train = False,
    download=False,
    transform = test_transform
)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size = 64,
    shuffle = False
)

# 同时加载原始图像用于可视化
test_dataset_raw = torchvision.datasets.CIFAR10(
    root='./data',
    train=False,
    download=False,
    transform=transforms.ToTensor()
)

print(f"测试集大小:{len(test_dataset)}")

# 4. 模型评估
print("\n3. 模型评估")

all_preds = []
all_labels = []
all_probs = []

with torch.no_grad():
    correct = 0
    total = 0
    for data,target in test_loader:
        data, target = data.to(device),target.to(device)
        output =  model(data)
        probabilities = torch.softmax(output, dim = 1)
        _,predicted = output.max(1)

        total += target.size(0)
        correct += predicted.eq(target).sum().item()

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(target.cpu().numpy())
        all_probs.extend(probabilities.cpu().numpy())

    accuracy = 100. * correct / total
    print(f"测试集准确率:{accuracy:.2f}%")


# 5. 混淆矩阵
print("\n4. 混淆矩阵")

cm = confusion_matrix(all_labels,all_preds)

plt.figure(figsize = (12,10))
sns.heatmap(cm,annot=True,fmt='d',cmap='Blues',
            xticklabels=classes_zh,yticklabels=classes_zh)
plt.xlabel('预测标签', fontsize=12)
plt.ylabel('真实标签', fontsize=12)
plt.title(f'混淆矩阵 (准确率: {accuracy:.2f}%)', fontsize=14)
plt.savefig('./confusion_matrix.png', dpi=150)
plt.show()

# 6. 分类报告
print("\n5. 分类报告")

print("\n各分类性能:")
print(classification_report(all_labels, all_preds, target_names=classes))

# 7. 各类别准确率
print("\n6. 各类别准确率")

class_correct = [0] * 10
class_total = [0] * 10

for true,pred in zip(all_labels,all_preds):
    class_total[true] += 1
    if pred == true:
        class_correct[true] += 1

plt.figure(figsize=(12, 6))
accuracies = [100. * class_correct[i] / class_total[i] for i in range(10)]
bars = plt.bar(classes_zh, accuracies)
plt.xlabel('类别', fontsize=12)
plt.ylabel('准确率 (%)', fontsize=12)
plt.title('各类别分类准确率', fontsize=14)
plt.xticks(rotation=45)

for bar, acc in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f'{acc:.1f}%', ha='center', va='bottom')

plt.ylim(0, 105)
plt.savefig('./class_accuracies.png', dpi=150)
plt.show()

print("\n各类别准确率:")
for i in range(10):
    print(f"  {classes_zh[i]}: {class_correct[i]}/{class_total[i]} ({accuracies[i]:.2f}%)")

# 8. 可视化正确和错误预测
print("\n7. 可视化预测结果")

def denormalize(img):
    """反标准化"""
    img = img * torch.tensor(std).view(3,1,1) + torch.tensor(mean).view(3,1,1)
    return img.clamp(0,1)

# 找出正确和错误的预测
correct_indices = [i for i,(t,p) in enumerate(zip(all_labels,all_preds)) if t==p]
wrong_indices = [i for i,(t,p) in enumerate(zip(all_labels,all_preds)) if t!=p]


print(f"正确预测数量: {len(correct_indices)}")
print(f"错误预测数量: {len(wrong_indices)}")

# 可视化正确预测
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
fig.suptitle('正确预测样本', fontsize=16)

for i, ax in enumerate(axes.flat):
    idx = correct_indices[i]
    img, true_label = test_dataset_raw[idx]
    pred_label = all_preds[idx]

    img_denorm = denormalize(img)
    ax.imshow(img_denorm.permute(1, 2, 0).numpy())
    ax.set_title(f'真实: {classes_zh[true_label]}\n预测: {classes_zh[pred_label]}',
                 color='green', fontsize=10)
    ax.axis('off')

plt.tight_layout()
plt.savefig('./correct_predictions.png', dpi=150)
plt.show()

# 可视化错误预测
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
fig.suptitle('错误预测样本', fontsize=16)

for i, ax in enumerate(axes.flat):
    if i < len(wrong_indices):
        idx = wrong_indices[i]
        img, true_label = test_dataset_raw[idx]
        pred_label = all_preds[idx]

        img_denorm = denormalize(img)
        ax.imshow(img_denorm.permute(1, 2, 0).numpy())
        ax.set_title(f'真实: {classes_zh[true_label]}\n预测: {classes_zh[pred_label]}',
                     color='red', fontsize=10)
    ax.axis('off')

plt.tight_layout()
plt.savefig('./wrong_predictions.png', dpi=150)
plt.show()

# 9. 保存评估结果
import json

evaluation_results = {
    'accuracy': accuracy,
    'total_correct': sum(class_correct),
    'total_samples': len(test_dataset),
    'class_accuracies': {classes_zh[i]: accuracies[i] for i in range(10)},
    'confusion_matrix': cm.tolist()
}

with open('./evaluation_results.json', 'w', encoding='utf-8') as f:
    json.dump(evaluation_results, f, indent=2, ensure_ascii=False)

print("\n评估结果已保存: learn8/evaluation_results.json")

print("\n CIFAR-10模型评估完成")



