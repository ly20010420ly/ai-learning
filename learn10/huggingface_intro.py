import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import torch
from transformers import pipeline,AutoTokenizer,AutoModel,AutoModelForSequenceClassification
import matplotlib.pyplot as plt
import numpy as np


# 设置中文
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("Hugging Face Transformers 入门")
print("=" * 60)

# 1. Transformers简介
print("\n1. Transformers库简介")

print("""
Hugging Face Transformers 是NLP领域最重要的库之一

核心概念：
  - Pipeline: 高级API，几行代码完成常见任务
  - Tokenizer: 将文本转换为模型可以理解的token ID
  - Model： 预训练模型（BERT,GPT,RoBERTa）
  - Trainer: 训练和微调API
  
支持的模型：
 - BERT： 双向编码器，擅长理解任务
 - GPT： 自回归解码器，擅长生成任务
 - T5: 编解码器，统一处理各种任务
 - 100+ 其他模型
""")

# 2. Pipeline快速上手
print("\n2. Pipeline快速上手 - 几行代码完成NLP任务")

# 情感分析
print("\n[1] 情感分析 （Sentiment Analysis）")
sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english",device="cuda")

texts = [
    "I love this movie! It's absolutely fantastic!",
    "This product is terrible, I want my money back.",
    "The weather is nice today.",
    "I'm not sure if I like it or not."
]

for text in texts:
    result = sentiment_pipeline(text)[0]
    print(f"文本：{text[:50]}...")
    print(f" 情感：{result['label']},置信度：{result['score']:.4f}")

# 文本生成
print("\n[2] 文本生成 (Text Generation)")

generator = pipeline("text-generation",model="distilgpt2")
prompt = "Once upon a time"
results = generator(prompt,max_length=50,num_return_sequences=2)
for i, result in enumerate(results):
        print(f"生成 {i+1}: {result['generated_text']}")

# 命名实体识别
print("\n[3] 命名实体识别 (Named Entity Recognition)")
ner_pipeline = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english", grouped_entities=True)
text = "Apple Inc. is headquartered in Cupertino, California. Tim Cook is the CEO."
results = ner_pipeline(text)
for entity in results:
    print(f"  实体: {entity['word']}, 类型: {entity['entity_group']}, 置信度: {entity['score']:.3f}")

# 问答系统
print("\n[4] 问答系统 (Question Answering)")
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
context = "The Eiffel Tower is located in Paris, France. It was built in 1889 and is 330 meters tall."
question = "When was the Eiffel Tower built?"
result = qa_pipeline(question=question, context=context)
print(f"问题: {question}")
print(f"回答: {result['answer']} (置信度: {result['score']:.3f})")

# 文本摘要
print("\n[5] 文本摘要 (Text Summarization)")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
long_text = """
The space station looks like an enormous jumble of solar arrays, modules, 
and struts. It is the result of decades of international cooperation between 
five space agencies: NASA, Roscosmos, ESA, JAXA, and CSA. The station serves 
as a microgravity and space environment research laboratory.
"""
summary = summarizer(long_text, max_length=30, min_length=10, do_sample=False)
print(f"原文: {long_text[:100]}...")
print(f"摘要: {summary[0]['summary_text']}")

# 3. 常用Pipeline任务总结
print("\n3. 常用Pipeline任务")

pipeline_tasks = {
    "feature_extraction":"特征提取",
    "fill_mask":"掩码填充",
    "ner":"命名实体识别",
    "question-answering":"问答",
    "sentiment-analysis":"情感分析",
    "summarization":"文本摘要",
    "text-generation":"文本生成",
    "translation":"翻译",
    "zero-shot-classfication":"零样本分类"
}

for task, desc in pipeline_tasks.items():
    print(f"  {task}: {desc}")

# 4. Tokenizer深入
print("\n4. Tokenizer详解")

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
text = "Hello, how are you today?"

# 编码
tokens = tokenizer.tokenize(text)
token_ids = tokenizer.encode(text)
print(f"原始文本：{text}")
print(f"Token列表：{tokens}")
print(f"Token IDs:{token_ids}")

# 带特殊Token的编码
encoded = tokenizer(text,max_length=10,padding=True,truncation=True,return_tensors="pt")  #padding不足10补齐（PAD=0），truncation超过截断
print(f"\n编码结果:")
print(f"  input_ids shape: {encoded['input_ids'].shape}")
print(f"  attention_mask: {encoded['attention_mask']}")

# 解码
decoded = tokenizer.decode(token_ids)
print(f"\n解码结果：{decoded}")

# 5. 模型架构
print("\n5. 模型架构")

# 加载模型
model = AutoModel.from_pretrained('distilbert-base-uncased')
print(f"模型类型：{type(model)}")
print(f"模型参数量：{sum(p.numel() for p in model.parameters()):,}")

# 查看模型结构
print(f"\n模型结构")
print(f" - 嵌入层：{model.embeddings}")
print(f" - Transformer层数：{len(model.transformer.layer)}")
print(f" - 隐藏层维度：{model.config.hidden_size}")
print(f" - 注意力头数：{model.config.num_attention_heads}")

# 6. 前向传播示例
print("\n6. 前向传播示例")

# 准备输入
text = "Hello World!"
encoded = tokenizer(text,return_tensors="pt")

# 前向传播
with torch.no_grad():
    outputs = model(**encoded)

print(f"输入文本：{text}")
print(f"输出形状：{outputs.last_hidden_state.shape}") #[batch, seq_len, hidden_size]
# print(f"池化输出形状：{outputs.pooler_output.shape}")

# 7. 保存模型信息
import json

model_info = {
    'name': 'distilbert-base-uncased',
    'num_parameters': sum(p.numel() for p in model.parameters()),
    'vocab_size': model.config.vocab_size,
    'hidden_size': model.config.hidden_size,
    'num_attention_heads': model.config.num_attention_heads,
    'num_hidden_layers': model.config.num_hidden_layers,
    'pipeline_tasks': pipeline_tasks
}

with open('./model_info.json', 'w') as f:
    json.dump(model_info, f, indent=2)

print("\n Hugging Face入门完成")


























