import torch
from transformers import BertTokenizer, BertModel, BertForSequenceClassification, BertForMaskedLM
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os

# 设置中文
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("BERT模型深入理解")
print("=" * 60)

# 创建保存目录
os.makedirs('day10', exist_ok=True)

# 1. 加载BERT模型和分词器
print("\n1. 加载BERT模型")

model_name = "bert-base-uncased"   #12层Transformer   768维特征   768维特征  不区分大小写
tokenizer = BertTokenizer.from_pretrained(model_name)
"""
from_pretrained
自动下载
vocab.txt：BERT词表
tokenizer_config.json
special_tokens_map.json
"""
model = BertModel.from_pretrained(model_name)

print(f"模型: {model_name}")
print(f"分词器词汇表大小: {tokenizer.vocab_size}")
print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

# 2. BERT输入格式
print("\n2. BERT输入格式")

text = "I love programming in Python!"
encoded = tokenizer(
    text,
    max_length=12,
    padding=True,
    truncation=True,
    return_tensors="pt"
)

print(f"原始文本: {text}")
print(f"Input IDs: {encoded['input_ids'][0]}")
print(f"Attention Mask: {encoded['attention_mask'][0]}")

# 解码查看tokens
tokens = tokenizer.convert_ids_to_tokens(encoded['input_ids'][0])
print(f"Tokens: {tokens}")

# 3. BERT前向传播
print("\n3. BERT前向传播")

with torch.no_grad():
    outputs = model(**encoded)

print(f"最后一层隐藏状态形状: {outputs.last_hidden_state.shape}")
print(f"池化输出形状: {outputs.pooler_output.shape}")

# 4. 可视化注意力权重（修正版）
print("\n4. BERT注意力可视化")

# 修正：重新加载带注意力输出的模型，避免参数冲突
attention_model = BertForSequenceClassification.from_pretrained(
    model_name,
    output_attentions=True
)
attention_model.eval()


def get_attention_weights(model, tokenizer, text):
    """获取BERT的注意力权重"""
    encoded = tokenizer(text, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**encoded)

    # 获取注意力权重 [层数, batch, heads, seq_len, seq_len]
    attentions = outputs.attentions
    return attentions, encoded


# 示例文本
text = "The cat sat on the mat"
attentions, encoded = get_attention_weights(attention_model, tokenizer, text)

print(f"注意力层数: {len(attentions)}")
print(f"每层注意力形状: {attentions[0].shape}")

# 可视化第一层的注意力
tokens = tokenizer.convert_ids_to_tokens(encoded['input_ids'][0])
attention_layer_0 = attentions[0][0, 0].numpy()  # 第一层，第一个head

plt.figure(figsize=(10, 8))
plt.imshow(attention_layer_0, cmap='viridis')
plt.colorbar(label='Attention Weight')
plt.xticks(range(len(tokens)), tokens, rotation=45)
plt.yticks(range(len(tokens)), tokens)
plt.title('BERT注意力权重可视化 (Layer 0, Head 0)')
plt.tight_layout()
plt.savefig('day10/bert_attention.png', dpi=150)
plt.show()

# 5. 词嵌入分析
print("\n5. 词嵌入分析")

# 获取词嵌入
word_embeddings = model.get_input_embeddings().weight

print(f"词嵌入矩阵形状: {word_embeddings.shape}")


# 查找相似词
def find_similar_words(word, top_k=5):
    """查找语义相似的词"""
    word_id = tokenizer.convert_tokens_to_ids(word)
    word_embedding = word_embeddings[word_id]

    # 修正1：使用 .detach().numpy() 分离梯度
    word_embedding_np = word_embedding.detach().numpy().reshape(1, -1)
    embeddings_np = word_embeddings.detach().numpy()

    # 计算相似度
    similarities = cosine_similarity(word_embedding_np, embeddings_np)[0]

    # 获取最相似的词（排除自身）
    top_indices = similarities.argsort()[-top_k - 1:-1][::-1]
    top_words = tokenizer.convert_ids_to_tokens(top_indices)

    print(f"与 '{word}' 最相似的词:")
    for w, sim in zip(top_words, similarities[top_indices]):
        print(f"  {w}: {sim:.4f}")


find_similar_words("good", top_k=5)
find_similar_words("computer", top_k=5)

# 6. 句子嵌入
print("\n6. 句子嵌入（池化方法）")

sentences = [
    "The cat is sleeping on the sofa.",
    "The dog is playing in the garden.",
    "I enjoy reading books about history.",
    "Artificial intelligence is transforming the world."
]


def get_sentence_embedding(text, pooling='cls'):
    """获取句子嵌入"""
    encoded = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**encoded)

    if pooling == 'cls':
        # 使用[CLS] token的表示
        embedding = outputs.pooler_output[0]
    elif pooling == 'mean':
        # 使用所有token的均值
        embedding = outputs.last_hidden_state[0].mean(dim=0)
    elif pooling == 'max':
        # 使用所有token的最大值
        embedding = outputs.last_hidden_state[0].max(dim=0)[0]

    return embedding


# 计算句子嵌入
embeddings = torch.stack([get_sentence_embedding(s, 'cls') for s in sentences])

# 计算相似度矩阵
similarity_matrix = cosine_similarity(embeddings.numpy())

plt.figure(figsize=(8, 6))
plt.imshow(similarity_matrix, cmap='coolwarm', vmin=-1, vmax=1)
plt.colorbar(label='Similarity')
plt.xticks(range(len(sentences)), [f"S{i + 1}" for i in range(len(sentences))])
plt.yticks(range(len(sentences)), [f"S{i + 1}" for i in range(len(sentences))])
plt.title('句子语义相似度矩阵')

# 添加数值标签
for i in range(len(sentences)):
    for j in range(len(sentences)):
        plt.text(j, i, f'{similarity_matrix[i, j]:.2f}',
                 ha='center', va='center')

plt.tight_layout()
plt.savefig('./sentence_similarity.png', dpi=150)
plt.show()

print("\n句子语义相似度:")
for i in range(len(sentences)):
    for j in range(i + 1, len(sentences)):
        print(f"  S{i + 1} vs S{j + 1}: {similarity_matrix[i, j]:.3f}")

# 7. BERT的Masked Language Model
print("\n7. Masked Language Model")

# 修正：单独加载MLM模型，避免与之前的模型冲突
mlm_model = BertForMaskedLM.from_pretrained(model_name)
mlm_model.eval()


def predict_masked_token(text, top_k=3):
    """预测[MASK]位置的token"""
    encoded = tokenizer(text, return_tensors="pt")

    with torch.no_grad():
        outputs = mlm_model(**encoded)
        logits = outputs.logits

    # 找到[MASK]的位置
    mask_index = (encoded['input_ids'][0] == tokenizer.mask_token_id).nonzero(as_tuple=True)[0].item()
    probs = torch.softmax(logits[0, mask_index], dim=-1)

    top_probs, top_indices = torch.topk(probs, top_k)

    print(f"输入: {text}")
    print(f"预测结果:")
    for prob, idx in zip(top_probs, top_indices):
        token = tokenizer.decode([idx])
        print(f"  {token}: {prob.item():.3f}")


# 测试MLM
predict_masked_token("The capital of France is [MASK].")
predict_masked_token("I love to read [MASK] books.")
predict_masked_token("The [MASK] is shining in the sky.")

# 8. 保存结果
print("\n8. 保存结果")
torch.save({
    'embeddings': word_embeddings,
    'model_config': model.config.to_dict()
}, './bert_embeddings.pth')

print("✅ BERT深入理解完成")