import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from datasets import Dataset
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt

print("=" * 60)
print("模型微调和保存")
print("=" * 60)

# 1. 创建自定义数据集
print("\n1. 创建自定义数据集")

# 模拟产品评论数据
reviews = [
    ("This product is amazing! Works perfectly.", 1),
    ("Terrible quality, broke after 2 days.", 0),
    ("Good value for money, would recommend.", 1),
    ("Not what I expected, very disappointed.", 0),
    ("Excellent customer service and fast shipping.", 1),
    ("The item arrived damaged, very unhappy.", 0),
    ("Works as described, satisfied with purchase.", 1),
    ("Completely useless, waste of money.", 0),
    ("Best purchase I've made this year!", 1),
    ("Poor build quality, not worth the price.", 0),
]

# 扩展数据集（复制多次）
reviews = reviews * 50  # 500条数据
texts = [r[0] for r in reviews]
labels = [r[1] for r in reviews]

print(f"数据集大小: {len(dataset)}")
print(f"正面样本: {sum(labels)}")
print(f"负面样本: {len(labels) - sum(labels)}")

# 2. 数据预处理
print("\n2. 数据预处理")

model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def preprocess_function(examples):
    return tokenizer(
        examples['text'],
        truncation=True,
        padding='max_length',
        max_length=128
    )

