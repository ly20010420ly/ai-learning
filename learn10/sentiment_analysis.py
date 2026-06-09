import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    pipeline
)
from datasets import load_dataset,Dataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import seaborn as sns
# 设置中文
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("情感分析实战")
print("=" * 60)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 1. 加载预训练情感分析模型
print("\n1. 加载预训练模型")

# 使用distilbert-base-uncased微调的情感分析模型
model_name = 'distilbert-base-uncased-finetuned-sst-2-english'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.to(device)

print(f"模型：{model_name}")
print(f"标签：{model.config.id2label}")

# 2. 使用Pipeline进行情感分析
print("\n2. Pipeline情感分析")

sentiment_pipeline = pipeline("sentiment-analysis",model = model_name)
test_texts = [
    "This movie is absolutely amazing! I loved every minute of it.",
    "The product broke after one day. Very disappointed.",
    "It's okay, nothing special but not terrible either.",
    "I can't believe how good this is! Highly recommend!",
    "Worst purchase ever. Complete waste of money.",
    "The service was excellent and the staff was friendly."
]


print("情感分析结果:")
for text in test_texts:
    result = sentiment_pipeline(text)[0]
    sentiment = "😊 POSITIVE" if result['label'] == 'POSITIVE' else "😞 NEGATIVE"
    print(f"{sentiment} ({result['score']:.3f}): {text[:60]}...")

# 3. 加载IMDB数据集
print("\n3. 加载IMDB电影评论数据集")
#尝试从datasets加载
imdb = load_dataset("imdb", split="train[:1000]")  # 只取前1000条用于演示
print(f"数据集大小: {len(imdb)}")
print(f"示例: {imdb[0]}")

# 4. 数据预处理
print("\n4. 数据预处理")

def preprocess_function(examples):
    """预处理函数"""
    return tokenizer(
        examples['text'],
        truncation=True,
        padding='max_length',
        max_length=128
    )

# 对数据集进行tokenizer
tokenized_imdb = imdb.map(preprocess_function,batched=True)
print(f"tokenized数据集形状：{len(tokenized_imdb)}")

# 5. 模型评估函数
print("\n5. 定义评估函数")

def compute_metrics(eval_pred):
    """计算评估指标"""
    predictions,labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    accuracy = accuracy_score(labels,predictions)
    f1 = f1_score(labels,predictions,average='weighted')

    return {
        "accuracy": accuracy,
        "f1": f1
    }

# 6. 微调模型（简化版，只训练少量epoch）
print("\n6. 微调模型")

# 划分训练集和数据集
train_test_split = tokenized_imdb.train_test_split(test_size=0.2)
train_dataset = train_test_split['train']
eval_dataset = train_test_split['test']

print(f"训练集大小: {len(train_dataset)}")
print(f"验证集大小: {len(eval_dataset)}")

# 训练参数
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=2,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    evaluation_strategy='epoch',  #每轮结束后评估模型在验证集上的表现
    save_strategy='no',
    logging_dir='./logs',
    report_to='none'
)

# 创建Trainer
trainer = Trainer(
    model = model,
    args = training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer = tokenizer,
    compute_metrics=compute_metrics
)

# 训练
print("开始训练...")
trainer.train()

# 7. 模型评估
print("\n7. 模型评估")

eval_result = trainer.evaluate()
print('评估结果')
for key,value in eval_result.items():
    print(f"  {key}: {value:.4f}")

# 8. 预测测试
print("\n8. 预测新评论")


def predict_sentiment(texts):
    """批量预测情感"""
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

    # 移动到GPU
    inputs = {
        k: v.to(device)
        for k, v in inputs.items()
    }

    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.softmax(outputs.logits, dim=-1)

    return predictions.cpu()


new_reviews = [
    "Absolutely incredible! Best film I've seen in years!",
    "Not worth the money. The quality is poor.",
    "It was okay. A bit slow but decent acting.",
    "I'm speechless. This changed my life!",
    "Complete disaster. Avoid at all costs."
]

predictions = predict_sentiment(new_reviews)

print("\n新评论预测:")
for review, pred in zip(new_reviews, predictions):
    pos_prob = pred[1].item()  # 正面概率
    neg_prob = pred[0].item()  # 负面概率
    sentiment = "POSITIVE" if pos_prob > neg_prob else "NEGATIVE"
    print(f"{sentiment} (正面: {pos_prob:.3f}, 负面: {neg_prob:.3f})")
    print(f"  {review[:60]}...")

# 9. 可视化预测分布
print("\n9. 可视化分析")

# 在验证集上预测
val_predictions = trainer.predict(eval_dataset)
pred_labels = np.argmax(val_predictions.predictions, axis=1)
true_labels = val_predictions.label_ids

# 混淆矩阵
cm = confusion_matrix(true_labels, pred_labels)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['NEGATIVE', 'POSITIVE'],
            yticklabels=['NEGATIVE', 'POSITIVE'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('情感分析混淆矩阵')
plt.savefig('./confusion_matrix_sentiment.png', dpi=150)
plt.show()

# 置信度分布
pos_confidences = []
neg_confidences = []

for pred, true in zip(val_predictions.predictions, true_labels):
    confidence = np.max(pred)
    if true == 1:
        pos_confidences.append(confidence)
    else:
        neg_confidences.append(confidence)

plt.figure(figsize=(10, 5))
plt.hist(pos_confidences, bins=20, alpha=0.5, label='POSITIVE', color='green')
plt.hist(neg_confidences, bins=20, alpha=0.5, label='NEGATIVE', color='red')
plt.xlabel('Confidence')
plt.ylabel('Frequency')
plt.title('预测置信度分布')
plt.legend()
plt.savefig('./confidence_distribution.png', dpi=150)
plt.show()

# 10. 保存微调模型
print("\n10. 保存模型")

model.save_pretrained('./finetuned_sentiment_model')
tokenizer.save_pretrained('./finetuned_sentiment_model')

print("模型已保存: learn10/finetuned_sentiment_model")

print("\n 情感分析实战完成")
