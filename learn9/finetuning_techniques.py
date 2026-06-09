import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

print("=" * 60)
print("微调技巧和优化")
print("=" * 60)

# 设置中文
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 1. 不同冻结策略
print("\n1. 不同冻结策略对比")

class FlexibleTransferLearning(nn.Module):
    """灵活的迁移学习模型"""
    def __init__(self,num_classes= 10,freeze_layers=0):
        super(FlexibleTransferLearning,self).__init__()

        self.backbone = torchvision.models.resnet18(pretrained=True)
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_features,num_classes)

        # 冻结指定数量的层
        if freeze_layers>0:
            layers = list(self.backbone.children())
            for i,layer in enumerate(layers[:freeze_layers]):
                for param in layer.parameters():
                    param.requires_grad = False

            print(f"冻结前 {freeze_layers}层")

    def forward(self,x):
        return self.backbone(x)

# 创建不同冻结策略的模型
for freeze in [0,2,4,6]:
    model = FlexibleTransferLearning(num_classes=10,freeze_layers=freeze)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"冻结层数：{freeze},可训练参数：{trainable_params:,}/{total_params:,}")

# 2. 分层学习率
print("\n2. 分层学习率")

def get_parameters(model,base_lr=0.001):
    """为不同层设置不同学习率"""
    param_groups = []

    # 分类头使用较高学习率
    fc_params = []
    backbone_params = []

    for name,param in model.named_parameters():
        if 'fc' in name:
            fc_params.append(param)
        else:
            backbone_params.append(param)

    param_groups.append({'params': fc_params, 'lr': base_lr})
    param_groups.append({'params': backbone_params, 'lr': base_lr/10})

    return param_groups

print("分层学习率策略:")
print("  - 骨干网络: lr/10 (保持预训练知识)")
print("  - 分类头: lr (快速适应新任务)")

# 3. 渐进式解冻
print("\n3. 渐进式解冻策略")

class ProgressiveUnfreezing(nn.Module):
    """渐进式冻结训练策略"""
    def __init__(self,model,num_stages=3):
        super(ProgressiveUnfreezing, self).__init__()
        self.model = model
        self.num_stages = num_stages
        self.stage=0

        # 获取所有需要冻结的层
        self.layers = self._get_layers()

    def _get_layers(self):
        """获取所有层"""
        layers = []
        for name,param in self.model.named_parameters():
            if 'fc' not in name:
                layers.append((name,param))
        return layers

    def unfreeze_next_stage(self):
        """冻结下一阶段"""
        if self.stage >= self.num_stages:
            return False

        #计算没阶段解冻的层数
        layers_per_stage = len(self.layers) // self.num_stages
        start = self.stage * layers_per_stage
        end = start + layers_per_stage

        for i in range(start,min(end,len(self.layers))):
            self.layers[i][1].requires_grad = True

        self.stage += 1
        print(f"Stage {self.stage}: 解冻了 {min(end, len(self.layers)) - start} 层")
        return True

progressive = ProgressiveUnfreezing(model, num_stages=4)
print("渐进式解冻策略:")
for i in range(4):
    progressive.unfreeze_next_stage()

# 4. 学习率预热
print("\n4. 学习率预热")

class WarmupScheduler:
    """学习率预热调度器"""
    def __init__(self,optimizer,warmup_epochs=5,target_lr=0.001):
        # super(WarmupScheduler,self).__init__()
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.target_lr = target_lr
        self.current_epoch = 0

    def step(self):
        self.current_epoch += 1
        if self.current_epoch <= self.warmup_epochs:
            #线性预热
            lr = self.target_lr * self.current_epoch / self.warmup_epochs
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            return lr
        return None

# 可视化学习率预热
warmup_epochs = 5
epochs = 30
lr_schedule = []

for epoch in range(epochs):
    if epoch < warmup_epochs:
        lr = 0.001 * (epoch+1) / warmup_epochs
    else:
        lr = 0.001 * (0.5 * (epoch - warmup_epochs) // 5)
    lr_schedule.append(lr)

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(range(epochs), lr_schedule)
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.title('预热 + 阶梯衰减')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
# 余弦退火
cosine_lr = [0.001 * (1 + np.cos(np.pi * e / epochs)) / 2 for e in range(epochs)]
plt.plot(range(epochs), cosine_lr)
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.title('预热 + 余弦退火')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('./lr_schedules.png', dpi=150)
plt.show()

# 5. 模型集成
print("\n5. 模型集成")

class ModelEnsemble:
    """模型集成"""
    def __init(self,models):
        self.models = models

    def predict(self,x):
        """投票预测"""
        predictions = []
        for model in self.models:
            with torch.no_grad():
                output = model(x)
                _,pred = output.max(1)
                predictions.append(pred)

        #投票
        predictions = torch.stack(predictions)
        final_pred,_ = torch.mode(predictions,dim=0)
        return final_pred

    def predict_proba(self,x):
        """概率平均"""
        probs = []
        for model in self.models:
            with torch.no_grad():
                output = model(x)
                prob = torch.softmax(output,dim=1)
                probs.append(prob)

        #平均概率
        return torch.mean(torch.stack(probs),dim=0)

print("模型集成方法:")
print("  - 投票法: 多数决定")
print("  - 平均法: 平均概率")
print("  - 加权法: 根据验证集性能加权")

# 6. 知识蒸馏
print("\n6. 知识蒸馏")

class DistillationLoss(nn.Module):
    """知识蒸馏损失"""
    def __init__(self,temperature=0.3,alpha=0.7):
        super(DistillationLoss, self).__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')

    def forward(self,student_output,teacher_output,target):
        # 硬标签损失
        hard_loss = self.ce_loss(student_output,target)

        # 软标签损失
        soft_student = torch.log_softmax(student_output / self.temperature, dim=1)
        soft_teacher = torch.softmax(teacher_output / self.temperature, dim=1)
        soft_loss = self.kl_loss(soft_student, soft_teacher) * (self.temperature ** 2)

        # 组合损失
        total_loss = self.alpha * hard_loss + (1 - self.alpha) * soft_loss
        return total_loss

print("知识蒸馏:")
print("  - 教师模型: 大模型/集成模型")
print("  - 学生模型: 小模型")
print("  - 温度参数: 控制softmax的平滑程度")

# 7. 对比学习简介
print("\n7. 对比学习简介")

print("""
对比学习（Contrastive Learning）:
  - 自监督学习的一种
  - 拉近正样本对，推远负样本对
  - 代表性方法: SimCLR, MoCo, BYOL

优势:
  - 不需要标注数据
  - 学习到更好的特征表示
  - 在小数据集上效果显著
""")

# 8. 保存微调技巧总结
with open('./finetuning_summary.md', 'w',encoding='UTF-8') as f:
    f.write("""
# 微调技巧总结

## 1. 冻结策略
- 数据少时：冻结大部分层，只训练分类头
- 数据多时：解冻更多层，甚至微调整个网络

## 2. 学习率策略
- 分类头：1e-3 ~ 1e-2
- 骨干网络：1e-5 ~ 1e-4（更小的学习率）

## 3. 数据增强
- 更强的数据增强可以防止过拟合
- 保持测试集增强一致

## 4. 正则化
- Dropout: 0.3 ~ 0.5
- Weight Decay: 1e-4

## 5. 训练技巧
- 使用学习率预热
- 渐进式解冻
- 早停法
- 模型集成

## 6. 常见问题
- 过拟合：减少训练轮数，增加Dropout
- 欠拟合：解冻更多层，增加学习率
- 训练不稳定：使用梯度裁剪
    """)

print("\n微调技巧总结已保存: learn9/finetuning_summary.md")

print("\n 微调技巧完成")










































