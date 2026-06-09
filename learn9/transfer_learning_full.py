import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.ao.pruning import scheduler
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import json
from tqdm import tqdm
import time

def main():
    # 设置中文
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False

    print("=" * 60)
    print("迁移学习完整实现")
    print("=" * 60)

    # 1. 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 2. 数据准备
    print("\n1. 数据准备")

    # CIFAR-10的均值和标准差
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]

    # 数据增强
    train_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(),   #随机水平翻转  RandomVerticalFlip 上下翻转
        transforms.RandomRotation(10),       #随机旋转   RandomCrop 随机裁剪  RandomResizedCrop裁剪加缩放
        transforms.ColorJitter(brightness=0.2,contrast=0.2),    #颜色扰动，亮度和对比度   RandomGrayscale随机变灰度图
        transforms.ToTensor(),
        transforms.Normalize(mean,std)
    ])

    #测试集
    test_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean,std)
    ])

    # 加载数据
    train_dataset = torchvision.datasets.CIFAR10(
        root = r'E:\python project\ai-learning\learn8\data',
        train = True,
        download = False,
        transform = train_transform
    )

    test_dataset = torchvision.datasets.CIFAR10(
        root = r'E:\python project\ai-learning\learn8\data',
        train = False,
        download = False,
        transform = test_transform
    )

    train_loader = DataLoader(
        dataset = train_dataset,
        batch_size = 64,
        shuffle = True,
        num_workers=2
    )

    test_loader = DataLoader(
        dataset = test_dataset,
        batch_size = 64,
        shuffle = False,
        num_workers=2
    )

    print(f"训练集: {len(train_dataset)} 样本")
    print(f"测试集: {len(test_dataset)} 样本")

    # 3. 创建迁移学习模型
    print("\n2. 创建迁移学习模型")

    class TransferLearningModel(nn.Module):
        """迁移学习模型（特征提取+新分类头）"""
        def __init__(self,num_classes=10,freeze_backbone=True):
            super(TransferLearningModel, self).__init__()

            #加载预训练RestNet8
            self.backbone = torchvision.models.resnet18(pretrained=True)

            # 冻结骨干网络
            if freeze_backbone:
                for param in self.backbone.parameters():
                    param.requires_grad = False

            # 获取特征维度
            num_features = self.backbone.fc.in_features

            # 替换分类头
            self.backbone.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(num_features, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, num_classes)
            )

            if freeze_backbone:
                self._freeze_backbone()

        def _freeze_backbone(self):
            """冻结主干网络"""
            # 除了最后的fc层，其他都冻结
            for name,param in self.backbone.named_parameters():
                if 'fc' not in name:
                    param.requires_grad=False

        def forward(self,x):
            return self.backbone(x)

    # 创建模型（特征提取模式）
    model_extract = TransferLearningModel(num_classes=10,freeze_backbone=True)
    model_finetune = TransferLearningModel(num_classes=10,freeze_backbone=False)

    print(f"特征提取模式参数：{sum(p.numel() for p in model_extract.parameters()):,}")
    print(f"   可训练参数：{sum(p.numel() for p in model_extract.parameters() if p.requires_grad):,}")
    print(f"微调模式参数：{sum(p.numel() for p in model_finetune.parameters()):,}")
    print(f"   可训练参数：{sum(p.numel() for p in model_finetune.parameters() if p.requires_grad)}")

    # 4.训练函数
    def train_model(model,train_loader,test_loader,epochs=20,lr=0.001):
        """训练模型"""
        model = model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

        #学习率调度
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        train_losses = []
        train_accs = []
        test_losses = []
        test_accs = []
        best_acc = 0

        for epoch in range(1,epochs+1):
            #训练
            model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0

            pbar = tqdm(train_loader,desc=f'Epoch {epoch/epochs}')
            for data,target in pbar:
                data,target = data.to(device),target.to(device)

                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = output.max(1)
                train_total += target.size(0)
                train_correct += predicted.eq(target).sum().item()

                pbar.set_postfix({'loss':f'{loss.item():.4f}',
                'acc':f'{100*train_correct/train_total:.2f}%'})

            train_loss /= len(train_loader)
            train_acc = 100.*train_correct / train_total
            train_losses.append(train_loss)
            train_accs.append(train_acc)

            #预测
            model.eval()
            test_loss = 0
            test_correct = 0
            test_total = 0

            with torch.no_grad():
                for data,target in test_loader:
                    data,target = data.to(device),target.to(device)
                    output = model(data)
                    loss = criterion(output,target)

                    test_loss += loss.item()
                    _,predicted = output.max(1)
                    test_total += target.size(0)
                    test_correct += predicted.eq(target).sum().item()

            test_loss /= len(test_loader)
            test_acc = 100.*test_correct / test_total
            test_losses.append(test_loss)
            test_accs.append(test_acc)

            scheduler.step()

            print(f'训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.2f}%')
            print(f'测试损失: {test_loss:.4f}, 测试准确率: {test_acc:.2f}%')

            if test_acc > best_acc:
                best_acc = test_acc
                torch.save(model.state_dict(),f'./best_model.pth')
                print(f'✓ 保存最佳模型 (准确率: {best_acc:.2f}%)')

        return train_losses, train_accs, test_losses, test_accs

    #5. 训练特征提取模式
    print("\n3. 训练特征提取模式")

    print("使用预训练ResNet18作为特征提取器...")
    extract_losses,extract_accs,extract_test_losses,extract_test_accs = train_model(
        model_finetune,train_loader,test_loader,epochs=10,lr=0.001
    )

    # 6. 训练微调模式
    print("\n4. 训练微调模式")

    print("微调整个网络...")
    finetune_losses, finetune_accs, finetune_test_losses, finetune_test_accs = train_model(
        model_finetune, train_loader, test_loader, epochs=15, lr=0.0001  # 更小的学习率
    )

    # 7. 对比结果可视化
    print("\n5. 对比训练结果")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 训练损失对比
    axes[0,0].plot(extract_losses,label='特征提取',linewidth=2)
    axes[0,0].plot(finetune_accs,label='微调',linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('训练损失对比')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 测试准确率对比
    axes[0, 1].plot(extract_test_accs, label='特征提取', linewidth=2)
    axes[0, 1].plot(finetune_test_accs, label='微调', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].set_title('测试准确率对比')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 最终准确率柱状图
    final_extract_acc = extract_test_accs[-1]
    final_finetune_acc = finetune_test_accs[-1]
    axes[1, 0].bar(['特征提取', '微调'], [final_extract_acc, final_finetune_acc])
    axes[1, 0].set_ylabel('Accuracy (%)')
    axes[1, 0].set_title('最终准确率对比')
    for i, acc in enumerate([final_extract_acc, final_finetune_acc]):
        axes[1, 0].text(i, acc + 1, f'{acc:.2f}%', ha='center')

    # 与从头训练的对比（使用第8天的结果）
    try:
        with open(r'E:\python project\ai-learning\learn8\training_info_ResNetCNN.json', 'r') as f:
            day8_info = json.load(f)
        scratch_acc = day8_info['best_acc']

        axes[1, 1].bar(['从头训练', '特征提取', '微调'],
                       [scratch_acc, final_extract_acc, final_finetune_acc])
        axes[1, 1].set_ylabel('Accuracy (%)')
        axes[1, 1].set_title('迁移学习 vs 从头训练')
        for i, acc in enumerate([scratch_acc, final_extract_acc, final_finetune_acc]):
            axes[1, 1].text(i, acc + 1, f'{acc:.2f}%', ha='center')
    except:
        axes[1, 1].text(0.5, 0.5, '第8天数据未找到\n无法对比',
                        ha='center', va='center')
        axes[1, 1].set_title('迁移学习 vs 从头训练')

    plt.tight_layout()
    plt.savefig('./transfer_learning_comparison.png', dpi=150)
    plt.show()

    # 8. 打印最终结果
    print("\n" + "=" * 60)
    print("最终结果对比")
    print("=" * 60)
    print(f"特征提取模式 - 最终准确率: {final_extract_acc:.2f}%")
    print(f"微调模式 - 最终准确率: {final_finetune_acc:.2f}%")
    print(f"提升幅度: {final_finetune_acc - final_extract_acc:.2f}%")

    print("\n 迁移学习训练完成")

if __name__ == '__main__':
    main()





























