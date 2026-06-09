import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import time
from tqdm import tqdm

# 设置中文
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("相似度搜索优化和评估")
print("=" * 60)

# 1. 加载模型
print("\n1. 加载模型")

model = SentenceTransformer('all-MiniLM-L6-v2')
print(f"模型加载成功，维度: {model.get_sentence_embedding_dimension()}")

# 2. 生成测试数据
print("\n2. 生成测试数据")

# 创建不同大小的文档集合
doc_sizes = [100, 500, 1000, 5000]
search_times = []

for size in doc_sizes:
    print(f"\n测试文档数量: {size}")

    # 生成随机文档
    documents = [f"This is document number {i} with some random content about AI."
                 for i in range(size)]

    # 编码文档
    start_time = time.time()
    doc_embeddings = model.encode(documents, show_progress_bar=False)
    encode_time = time.time() - start_time
    print(f"  编码时间: {encode_time:.2f}秒")

    # 查询
    query = "What is artificial intelligence?"
    query_embedding = model.encode([query])

    # 线性搜索
    start_time = time.time()
    similarities = cosine_similarity(query_embedding, doc_embeddings)[0]
    top_indices = np.argsort(similarities)[-5:][::-1]
    linear_time = time.time() - start_time
    print(f"  线性搜索时间: {linear_time:.4f}秒")

    search_times.append({
        'size': size,
        'linear_time': linear_time,
        'encode_time': encode_time
    })

# 可视化搜索时间
plt.figure(figsize=(10, 6))
sizes = [t['size'] for t in search_times]
linear_times = [t['linear_time'] for t in search_times]

plt.plot(sizes, linear_times, 'o-', linewidth=2, label='线性搜索')
plt.xlabel('文档数量')
plt.ylabel('搜索时间 (秒)')
plt.title('文档数量 vs 搜索时间')
plt.grid(True, alpha=0.3)
plt.legend()
plt.savefig('./search_time_analysis.png', dpi=150)
plt.show()

# 3. 不同相似度度量对比
print("\n3. 不同相似度度量对比")

sentences = [
    "Python is a programming language.",
    "Python is great for data science.",
    "Java is another programming language.",
    "The weather is nice today.",
]

embeddings = model.encode(sentences)

# 计算不同度量
cosine_sim = cosine_similarity(embeddings)
euclidean_dist = np.sqrt(np.sum((embeddings[:, None] - embeddings) ** 2, axis=-1))

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].imshow(cosine_sim, cmap='coolwarm', vmin=0, vmax=1)
axes[0].set_title('余弦相似度')
axes[0].set_xlabel('句子索引')
axes[0].set_ylabel('句子索引')
for i in range(len(sentences)):
    for j in range(len(sentences)):
        axes[0].text(j, i, f'{cosine_sim[i, j]:.2f}',
                     ha='center', va='center')

axes[1].imshow(euclidean_dist, cmap='viridis')
axes[1].set_title('欧氏距离')
axes[1].set_xlabel('句子索引')
axes[1].set_ylabel('句子索引')
for i in range(len(sentences)):
    for j in range(len(sentences)):
        axes[1].text(j, i, f'{euclidean_dist[i, j]:.2f}',
                     ha='center', va='center')

plt.tight_layout()
plt.savefig('./similarity_metrics.png', dpi=150)
plt.show()

print("\n相似度度量说明:")
print("  余弦相似度: 值越大越相似 (范围: -1 到 1)")
print("  欧氏距离: 值越小越相似 (范围: 0 到 ∞)")

# 4. 阈值调优
print("\n4. 相似度阈值调优")

# 创建正样本和负样本
positive_pairs = [
    ("Python is great", "Python is awesome"),
    ("Machine learning", "ML algorithms"),
    ("Data science", "Data analysis"),
]

negative_pairs = [
    ("Python programming", "The weather is sunny"),
    ("Deep learning", "Cooking recipes"),
    ("Computer vision", "Historical events"),
]

positive_scores = []
for s1, s2 in positive_pairs:
    emb1 = model.encode([s1])
    emb2 = model.encode([s2])
    score = cosine_similarity(emb1, emb2)[0][0]
    positive_scores.append(score)

negative_scores = []
for s1, s2 in negative_pairs:
    emb1 = model.encode([s1])
    emb2 = model.encode([s2])
    score = cosine_similarity(emb1, emb2)[0][0]
    negative_scores.append(score)

# 绘制分数分布
plt.figure(figsize=(10, 6))
plt.hist(positive_scores, bins=10, alpha=0.5, label='正样本对', color='green')
plt.hist(negative_scores, bins=10, alpha=0.5, label='负样本对', color='red')
plt.xlabel('余弦相似度')
plt.ylabel('频数')
plt.title('正负样本对的相似度分布')
plt.legend()
plt.savefig('./threshold_analysis.png', dpi=150)
plt.show()

print("正样本对相似度:")
for i, (pair, score) in enumerate(zip(positive_pairs, positive_scores)):
    print(f"  {i + 1}. '{pair[0]}' vs '{pair[1]}': {score:.3f}")

print("\n负样本对相似度:")
for i, (pair, score) in enumerate(zip(negative_pairs, negative_scores)):
    print(f"  {i + 1}. '{pair[0]}' vs '{pair[1]}': {score:.3f}")

# 5. 批量查询优化
print("\n5. 批量查询优化")


def batch_search(queries, doc_embeddings, batch_size=32):
    """批量查询优化"""
    all_results = []

    for i in range(0, len(queries), batch_size):
        batch_queries = queries[i:i + batch_size]
        batch_embeddings = model.encode(batch_queries)
        batch_similarities = cosine_similarity(batch_embeddings, doc_embeddings)
        all_results.extend(batch_similarities)

    return all_results


# 测试
test_queries = [f"Query {i} about AI" for i in range(100)]
doc_embeddings = model.encode([f"Document {i}" for i in range(1000)])

# 单条查询
start = time.time()
for q in test_queries[:10]:
    q_emb = model.encode([q])
    _ = cosine_similarity(q_emb, doc_embeddings)
single_time = time.time() - start

# 批量查询
start = time.time()
_ = batch_search(test_queries[:10], doc_embeddings)
batch_time = time.time() - start

print(f"单条查询10次: {single_time:.4f}秒")
print(f"批量查询10条: {batch_time:.4f}秒")
print(f"加速比: {single_time / batch_time:.2f}x")

# 6. 缓存策略
print("\n6. 缓存策略")


class CachedRetriever:
    """带缓存的检索器"""

    def __init__(self, model):
        self.model = model
        self.cache = {}

    def encode(self, text):
        if text in self.cache:
            return self.cache[text]
        embedding = self.model.encode([text])[0]
        self.cache[text] = embedding
        return embedding

    def get_cache_stats(self):
        return f"缓存大小: {len(self.cache)}"


cached_retriever = CachedRetriever(model)

# 测试缓存效果
test_texts = ["What is AI?", "What is ML?", "What is AI?"] * 20

start = time.time()
for text in test_texts:
    _ = cached_retriever.encode(text)
cached_time = time.time() - start

print(f"带缓存的编码时间: {cached_time:.4f}秒")
print(f"{cached_retriever.get_cache_stats()}")

# 7. 优化建议总结
print("\n7. 相似度搜索优化建议")

optimization_tips = """
┌─────────────────────────────────────────────────────────────────┐
│ 优化策略              │ 效果                   │ 适用场景        │
├─────────────────────────────────────────────────────────────────┤
│ 使用索引（FAISS）     │ 100-1000倍加速         │ 大规模检索       │
│ 批量查询              │ 2-5倍加速              │ 多查询场景       │
│ 缓存查询结果          │ 1.5-3倍加速（重复查询）│ 热点查询         │
│ 降低嵌入维度          │ 1.2-2倍加速            │ 精度要求不高     │
│ 使用更轻量模型        │ 2-5倍加速              │ 实时性要求高     │
│ 预计算并保存嵌入      │ 避免重复计算           │ 固定文档集       │
│ 使用近似最近邻(ANN)   │ 10-100倍加速           │ 容忍少量误差     │
└─────────────────────────────────────────────────────────────────┘
"""
print(optimization_tips)

# 8. 保存优化分析
optimization_results = {
    'search_times': search_times,
    'positive_scores': positive_scores,
    'negative_scores': negative_scores,
    'batch_analysis': {
        'single_query_time': single_time,
        'batch_query_time': batch_time,
        'speedup': single_time / batch_time
    }
}

import json

with open('./optimization_results.json', 'w') as f:
    json.dump(optimization_results, f, indent=2, default=float)

print("\n✅ 相似度搜索优化和评估完成")