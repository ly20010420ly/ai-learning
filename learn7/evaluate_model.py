import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

print("=" * 60)
print("模型评估和可视化")
print("=" * 60)

# 1. 设置设备
print("\n1. 设置设备")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 2. 加载模型
print("\n2. 加载模型")

from cnn_model import CNN

model = CNN().to(device)
model.load_state_dict(torch.load('./learn7/best_model.pth', map_location=device))
model.eval()

print("模型加载成功")

# 3. 加载测试数据
print("\n3. 加载测试数据")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

test_dataset = torchvision.datasets.MNIST(
    root='./learn7/data',
    train=False,
    download=False,
    transform=transform
)

test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)
print(f"测试集大小: {len(test_dataset)}")


# 4. 模型评估
print("\n4. 模型评估")

all_preds = []
all_labels = []

with torch.no_grad():
    correct = 0
    total = 0
    for data,target in test_loader:
        data,target = data.to(device), target.to(device)
        output = model(data)
        _,predicted = output.max(1)   #取最大值索引
        total += target.size(0)   #样本数累加
        correct += predicted.eq(target).sum().item()  #预测正确数量

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(target.cpu().numpy())

accuracy = 100. * correct / total
print(f"测试集准确率: {accuracy:.2f}%")


# 5. 混淆矩阵
print("\n5. 混淆矩阵")

cm = confusion_matrix(all_labels,all_preds)

plt.figure(figsize=[10,8])
sns.heatmap(cm,annot=True,fmt='d',cmap='Blues',
            xticklabels=range(10),yticklabels=range(10))
plt.xlabel('预测标签')
plt.ylabel('真实标签')
plt.title(f'混淆矩阵 (准确率: {accuracy:.2f}%)')
plt.savefig('./confusion_matrix.png', dpi=150)
plt.show()


# 6. 分类报告
print("\n6. 分类报告")
#生成分类模型评估报告（Classification Report）
"""
每个类别的：
- Precision（精确率）
- Recall（召回率）
- F1-score
- Support（样本数）
"""
print(classification_report(all_labels, all_preds,
                            target_names=[str(i) for i in range(10)]))


# 7. 错误分析
print("\n7. 错误分析")

#找出错误预测的样本
errors = []
for i,(true,pred) in enumerate(zip(all_labels,all_preds)):
    if true != pred:
        errors.append((i,true,pred))

print(f"错误预测总数: {len(errors)}")
print(f"错误率: {len(errors)/len(test_dataset)*100:.2f}%")

#统计每类错误
error_by_class = {}
for _,true,pred in errors:
    key = f"{true}->{pred}"
    error_by_class[key] = error_by_class.get(key,0) + 1

print("\n最常见的错误类型:")
sorted_errors = sorted(error_by_class.items(), key=lambda x:x[1], reverse=True)
for i, (error_type, count) in enumerate(sorted_errors[:10]):
    print(f"  {error_type}: {count} 次")


# 8. 可视化错误样本
print("\n8. 可视化错误样本")

fig, axes = plt.subplots(2, 5, figsize=(15, 6))
fig.suptitle('错误预测样本分析', fontsize=16)

# 获取原始图像（未归一化）
test_dataset_raw = torchvision.datasets.MNIST(
    root='./learn7/data',
    train=False,
    download=False,
    transform=transforms.ToTensor()  # 只转换，不归一化
)

for i, ax in enumerate(axes.flat):
    if i < len(errors):
        idx, true, pred = errors[i]
        img = test_dataset_raw[idx][0].numpy().squeeze()

        ax.imshow(img, cmap='gray')
        ax.set_title(f'真实: {true}, 预测: {pred}', fontsize=10)
        ax.axis('off')

plt.tight_layout()
plt.savefig('./error_samples.png', dpi=150)
plt.show()


# 9. 每个数字的准确率
print("\n9. 每个数字的分类准确率")

class_correct = [0] * 10
class_total = [0] * 10

for true, pred in zip(all_labels, all_preds):
    class_total[true] += 1
    if true == pred:
        class_correct[true] += 1

plt.figure(figsize=(10, 6))
classes = range(10)
accuracies = [100. * class_correct[i] / class_total[i] for i in classes]
bars = plt.bar(classes, accuracies)
plt.xlabel('数字类别')
plt.ylabel('准确率 (%)')
plt.title('各类别分类准确率')

# 添加数值标签
for bar, acc in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f'{acc:.1f}%', ha='center', va='bottom')

plt.xticks(classes)
plt.ylim(0, 105)
plt.savefig('./class_accuracies.png', dpi=150)
plt.show()

print("\n各类别准确率:")
for i in range(10):
    print(f"  数字 {i}: {class_correct[i]}/{class_total[i]} ({accuracies[i]:.2f}%)")

# 10. 保存评估结果
import json

evaluation_results = {
    'accuracy': accuracy,
    'total_errors': len(errors),
    'error_rate': len(errors)/len(test_dataset)*100,
    'class_accuracies': {str(i): accuracies[i] for i in range(10)},
    'confusion_matrix': cm.tolist()
}

with open('./evaluation_results.json', 'w') as f:
    json.dump(evaluation_results, f, indent=2)

print("\n评估结果已保存: learn/evaluation_results.json")

print("\n 模型评估完成")