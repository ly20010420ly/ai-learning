import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from networkx.algorithms import similarity
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# 设置中文
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("特征提取方法")
print("=" * 60)

# 1. 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 2. 加载预训练模型
print("\n1. 加载预训练ResNet18")

resnet18 = torchvision.models.resnet18(pretrained=True)
resnet18 = resnet18.to(device)
resnet18.eval()

print(f"ResNet18加载成功")

# 3. 创建特征提取器
print("\n2. 创建特征提取器")

class FeatureExtractor(nn.Module):
    """从预训练模型提取特征"""
    def __init__(self,model,layer_name='avgpool'):
        super(FeatureExtractor,self).__init__()
        self.model = model
        self.layer_name = layer_name
        self.features = None

        #注册钩子
        self._register_hook()

    def _register_hook(self):
        def hook(moudle,input,output):
            self.features = output.detach()

        # 获取指导层
        for name,module in self.model.named_modules():
            if name == self.layer_name:
                module.register_forward_hook(hook)
                print(f"钩子已注册到：{name}")

    def forward(self,x):
        _ = self.model(x)
        return self.features

# 创建特征提取器
feature_extractor = FeatureExtractor(resnet18,'avgpool')
print(f"特征提取器创建成功")

# 4. 准备数据
print("\n3. 准备测试数据")

# CIFAR-10的标准化参数
mean = [0.4914, 0.4822, 0.4465]
std = [0.2023, 0.1994, 0.2010]

# 需要将图像resize到224*224(Resnet输入大小)
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean,std)
])

# 加载数据集
test_dataset = torchvision.datasets.CIFAR10(root=r'E:\python project\ai-learning\learn8\data',train=False,download=False,transform=transform)

test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=32,shuffle=False)

print(f"测试集大小: {len(test_dataset)}")

# 5. 提取特征
print("\n4. 提取特征")

features_list = []
labels_list = []

with torch.no_grad():
    for images,labels in test_loader:
        images = images.to(device)
        features = feature_extractor(images)

        #展平特征
        features = features.view(features.size(0),-1)
        features_list.append(features.cpu())
        labels_list.append(labels)

features = torch.cat(features_list,dim=0)
labels = torch.cat(labels_list,dim=0)

print(f"提取的特征形状: {features.shape}")
print(f"特征维度: {features.shape[1]}")

# 6. 特征统计
print("\n5. 特征统计")

# 计算特征均值和方差
feature_mean = features.mean(dim=0)
feature_std = features.std(dim=0)

print(f"特征均值范围: [{feature_mean.min():.4f}, {feature_mean.max():.4f}]")
print(f"特征标准差范围: [{feature_std.min():.4f}, {feature_std.max():.4f}]")
print(f"零均值特征比例: {(feature_mean.abs() < 0.01).float().mean():.2%}")

# 7. 可视化特征分布（使用PCA降维）
print("\n6. 可视化特征分布")

# 随机选择2000各样本进行可视化
n_samples = 2000
indices = np.random.choice(len(features),n_samples,replace=False)
sample_features = features[indices]
sample_labels = labels[indices]

#PCA降维
pca = PCA(n_components=2)
features_pca = pca.fit_transform(sample_features)

print(f"PCA解释方差比：{pca.explained_variance_ratio_}")
print(f"总解释方差：{pca.explained_variance_ratio_.sum():.3f}")

# 绘制PCA可视化
plt.figure(figsize=(15,2))

plt.subplot(1,2,1)
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']
colors = plt.cm.tab10(np.linspace(0, 1, 10))

for i in range(10):
    mask = sample_labels == i
    plt.scatter(features_pca[mask,0],features_pca[mask,1],
                c=[colors[i]],label=classes[i],alpha=0.5,s=10)

plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
plt.title('特征PCA可视化')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# t-SNE可视化
plt.subplot(1, 2, 2)
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
features_tsne = tsne.fit_transform(sample_features)

for i in range(10):
    mask = sample_labels == i
    plt.scatter(features_tsne[mask, 0], features_tsne[mask, 1],
                c=[colors[i]], label=classes[i], alpha=0.5, s=10)

plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.title('特征t-SNE可视化')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.savefig('./feature_visualization.png', dpi=150)
plt.show()

# 8. 特征相似度分析
print("\n7. 特征相似度分析")

# 计算类别原型（平均特征）
class_prototypes = []
for i in range(10):
    class_features = features[labels == i]
    prototype = class_features.mean(dim=0)
    class_prototypes.append(prototype)

class_prototypes = torch.stack(class_prototypes)

# 计算类别间相似度
similarity = torch.nn.functional.cosine_similarity(
    class_prototypes.unsqueeze(1),
    class_prototypes.unsqueeze(0),
    dim=2
)

plt.figure(figsize=(10, 8))
plt.imshow(similarity, cmap='coolwarm', vmin=0, vmax=1)
plt.colorbar(label='Cosine Similarity')
plt.xticks(range(10), classes, rotation=45)
plt.yticks(range(10), classes)
plt.title('类别特征相似度矩阵')
for i in range(10):
    for j in range(10):
        plt.text(j, i, f'{similarity[i, j]:.2f}',
                ha='center', va='center')
plt.tight_layout()
plt.savefig('./feature_similarity.png', dpi=150)
plt.show()

print("\n最相似的类别对:")
similarity_nodiag = similarity.clone()
similarity_nodiag.fill_diagonal_(0)
max_idx = similarity_nodiag.argmax()
i, j = max_idx // 10, max_idx % 10
print(f"  {classes[i]} - {classes[j]}: {similarity[i, j]:.3f}")

print("\n最不相似的类别对:")
min_idx = similarity_nodiag.argmin()
i, j = min_idx // 10, min_idx % 10
print(f"  {classes[i]} - {classes[j]}: {similarity[i, j]:.3f}")

# 9. 保存特征
torch.save({
    'features': features,
    'labels': labels,
    'class_prototypes': class_prototypes
}, '../data and pt/learn9/extracted_features.pth')

print("\n特征已保存: ./extracted_features.pth")

print("\n 特征提取完成")





















