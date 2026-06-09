import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
def main():
    # 设置中文
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False

    print("=" * 60)
    print("数据增强效果对比实验")
    print("=" * 60)

    # 1. 定义不同的数据增强策略
    print("\n1. 定义数据增强策略")

    strategies = {
        '无增强': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010))
        ]),

        '水平翻转': transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010))
        ]),

        '随机裁剪': transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010))
        ]),

        '完整增强': transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010))
        ])
    }

    print("定义了4种增强策略")

    # 2. 可视化不同增强策略的效果
    print("\n2. 可视化增强效果")

    # 加载原始图像
    raw_dataset = torchvision.datasets.CIFAR10(
        root='./data',
        train=True,
        download=False,
        transform=None)

    # 选择5个样本
    sample_indices = [0, 100, 500, 1000, 5000]

    fig, axes = plt.subplots(len(sample_indices), len(strategies),
                             figsize=(15, 12))
    fig.suptitle('数据增强策略对比', fontsize=16)

    for i, idx in enumerate(sample_indices):
        original_img = raw_dataset[idx][0]

        for j, (strategy_name, transform) in enumerate(strategies.items()):
            torch.manual_seed(42)
            aug_img = transform(original_img)

            # 反标准化
            mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
            std = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1)
            display_img = aug_img * std + mean
            display_img = display_img.clamp(0, 1)

            ax = axes[i, j]
            ax.imshow(display_img.permute(1, 2, 0).numpy())
            if i == 0:
                ax.set_title(strategy_name, fontsize=12)
            if j == 0:
                ax.set_ylabel(f'样本 {i + 1}', fontsize=10)
            ax.axis('off')

    plt.tight_layout()
    plt.savefig('./augmentation_strategies.png', dpi=150)
    plt.show()

    # 3. 训练简单模型对比（快速实验）
    print("\n3. 快速训练对比实验")

    import torch.nn as nn
    import torch.optim as optim


    class SimpleCNN(nn.Module):
        def __init__(self):
            super(SimpleCNN, self).__init__()
            self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
            self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
            self.pool = nn.MaxPool2d(2, 2)
            self.fc1 = nn.Linear(64 * 8 * 8, 128)
            self.fc2 = nn.Linear(128, 10)

        def forward(self, x):
            x = self.pool(torch.relu(self.conv1(x)))
            x = self.pool(torch.relu(self.conv2(x)))
            x = x.view(x.size(0), -1)
            x = torch.relu(self.fc1(x))
            x = self.fc2(x)
            return x


    def quick_train(strategy_name, transform, epochs=5):
        """快速训练评估"""
        print(f"\n训练: {strategy_name}")

        # 加载数据
        train_dataset = torchvision.datasets.CIFAR10(
            root='./data',
            train=True,
            transform=transform,
            download=False
        )

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=64, shuffle=True, num_workers=2
        )

        test_dataset = torchvision.datasets.CIFAR10(
            root='./data',
            train=False,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.2023, 0.1994, 0.2010))
            ]),
            download=False
        )

        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=64, shuffle=False, num_workers=2
        )

        # 训练
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = SimpleCNN().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        accuracies = []
        for epoch in range(epochs):
            model.train()
            for data, target in train_loader:
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

            # 评估
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    _, predicted = output.max(1)
                    total += target.size(0)
                    correct += predicted.eq(target).sum().item()

            acc = 100. * correct / total
            accuracies.append(acc)
            print(f"  Epoch {epoch + 1}: {acc:.2f}%")

        return accuracies


    # 4. 运行快速实验（可选，如果时间不够可以注释）
    print("\n4. 运行快速实验（需要约10分钟）")

    run_experiment = input("是否运行快速实验？(y/n): ").lower() == 'y'

    if run_experiment:
        results = {}
        for name, transform in strategies.items():
            accs = quick_train(name, transform, epochs=3)
            results[name] = accs

        # 绘制对比图
        plt.figure(figsize=(12, 6))
        for name, accs in results.items():
            plt.plot(range(1, len(accs) + 1), accs, marker='o', label=name)

        plt.xlabel('Epoch')
        plt.ylabel('准确率 (%)')
        plt.title('不同数据增强策略的效果对比')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('./augmentation_comparison_results.png', dpi=150)
        plt.show()

        # 打印最终结果
        print("\n最终准确率对比:")
        for name, accs in results.items():
            print(f"  {name}: {accs[-1]:.2f}%")
    else:
        print("跳过快速实验，继续进行")

    # 5. 数据增强建议
    print("\n5. 数据增强建议")

    recommendations = """
    数据增强最佳实践:
    
    1. 基本增强（适用于大多数情况）:
       - 随机水平翻转
       - 随机裁剪
       - 颜色抖动（轻微）
    
    2. 针对CIFAR-10:
       - 翻转: 帮助模型学习对称性
       - 裁剪: 增加位置不变性
       - 注意: 不要过度增强，会损害性能
    
    3. 针对小数据集:
       - 使用更强的增强
       - 混合增强策略
       - 使用CutMix或MixUp
    
    4. 注意事项:
       - 测试集不要使用增强
       - 保持标签一致性
       - 验证增强后的图像质量
    """

    print(recommendations)

    print("\n 数据增强对比实验完成")

if __name__ == '__main__':
    main()