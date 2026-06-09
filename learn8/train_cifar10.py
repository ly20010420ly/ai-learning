import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

def main():
    # 设置中文
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False

    print("=" * 60)
    print("CIFAR-10 模型训练")
    print("=" * 60)

    # 1. 设置设备
    print("\n1. 设置设备")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # 2. 数据加载
    print("\n2. 数据加载")

    mean = [0.4914,0.4822,0.4465]
    std = [0.2023,0.1994,0.2010]

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean,std)
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean,std)
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=train_transform
        )
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data',
        train = False,
        download=False,
        transform = test_transform
    )

    batch_size = 128
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size,
                             shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=2, pin_memory=True)

    print(f"训练集: {len(train_dataset)} 样本, {len(train_loader)} 批次")
    print(f"测试集: {len(test_dataset)} 样本, {len(test_loader)} 批次")

    # 3. 创建模型
    print("\n3. 创建模型")

    from cifar10_cnn import BasicCNN, ResNetCNN

    # 选择模型
    use_resnet = True  # 可以切换为False使用基础CNN

    if use_resnet:
        model = ResNetCNN().to(device)
        model_name = "ResNetCNN"
    else:
        model = BasicCNN().to(device)
        model_name = "BasicCNN"

    print(f"模型: {model_name}")
    print(f"参数数量: {sum(p.numel() for p in model.parameters()):,}")


    # 4. 定义损失函数和优化器
    print("\n4. 定义训练配置")
    """
    模型如何学习
    如何更新参数
    如何提高精度
    """
    criterion = nn.CrossEntropyLoss()  #交叉熵损失函数 Softmax + NLLLoss

    # 使用SGD + Momentum（通常比Adam效果好）
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9,weight_decay=5e-4)
    #学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=50)

    print(f"损失函数: CrossEntropyLoss")
    print(f"优化器: SGD (lr=0.01, momentum=0.9, weight_decay=5e-4)")
    print(f"学习率调度: CosineAnnealingLR")

    # 5. 训练函数
    print("\n5. 开始训练")

    def train_epoch(model,loader,criterion,optimizer,device):
        model.train()
        running_loss = 0.0   #累计loss
        correct = 0          #预测正确数
        total = 0            #总样本数

        pbar = tqdm(loader,desc='训练')  #显示训练进度条
        for data, target in pbar:
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _,predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()   #统计正确数

            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100. * correct / total:.2f}%'})  #实时更新进度条

        epoch_loss = running_loss / len(loader)
        epoch_acc = 100. * correct / total
        return epoch_loss, epoch_acc

    def validate(model,loader,ceiterion,device):
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for data,target in tqdm(loader,desc='验证'):
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                test_loss += loss.item()
                _,predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()

        test_loss = test_loss / len(loader)
        test_acc = 100. * correct / total
        return test_loss, test_acc


    # 6. 训练循环
    num_epochs = 50
    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []
    best_acc = 0

    start_time = time.time()

    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        print("-" * 40)

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = validate(model, test_loader, criterion, device)

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        print(f"训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.2f}%")
        print(f"验证损失: {test_loss:.4f}, 验证准确率: {test_acc:.2f}%")
        print(f"学习率: {current_lr:.6f}")

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), f'./best_{model_name}.pth')
            print(f"✓ 保存最佳模型 (准确率: {best_acc:.2f}%)")

        # 早停
        if best_acc > 85 and epoch > 30:
            print(f"已达到目标准确率，提前结束训练")
            break

    training_time = time.time() - start_time
    print(f"\n训练完成！总用时: {training_time:.2f} 秒 ({training_time / 60:.2f} 分钟)")
    print(f"最佳验证准确率: {best_acc:.2f}%")

    # 7. 可视化训练过程
    print("\n6. 可视化训练过程")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 损失曲线
    axes[0, 0].plot(train_losses, label='训练损失', linewidth=2)
    axes[0, 0].plot(test_losses, label='验证损失', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('损失曲线')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 准确率曲线
    axes[0, 1].plot(train_accs, label='训练准确率', linewidth=2)
    axes[0, 1].plot(test_accs, label='验证准确率', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].set_title('准确率曲线')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 最终准确率
    epochs_range = range(1, len(train_accs) + 1)
    axes[1, 0].bar(epochs_range[-10:], train_accs[-10:], alpha=0.7, label='训练')
    axes[1, 0].bar(epochs_range[-10:], test_accs[-10:], alpha=0.7, label='验证')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy (%)')
    axes[1, 0].set_title('最后10个epoch的准确率')
    axes[1, 0].legend()

    # 损失差（过拟合指示）
    loss_diff = [t - v for t, v in zip(train_losses, test_losses)]
    axes[1, 1].plot(loss_diff, linewidth=2, color='orange')
    axes[1, 1].axhline(y=0, color='r', linestyle='--')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss Difference (Train - Val)')
    axes[1, 1].set_title('过拟合检测（正值表示过拟合）')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'./training_history_{model_name}.png', dpi=150)
    plt.show()

    # 8. 保存训练记录
    import json

    training_info = {
        'model_name': model_name,
        'num_epochs': len(train_accs),
        'best_acc': best_acc,
        'training_time': training_time,
        'train_losses': train_losses,
        'train_accs': train_accs,
        'test_losses': test_losses,
        'test_accs': test_accs
    }

    with open(f'./training_info_{model_name}.json', 'w') as f:
        json.dump(training_info, f, indent=2)

    print(f"\n训练记录已保存: learn8/training_info_{model_name}.json")

    print("\n CIFAR-10训练完成")

if __name__ == '__main__':
    main()


