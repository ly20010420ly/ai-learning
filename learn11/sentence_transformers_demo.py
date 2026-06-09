from sentence_transformers import SentenceTransformer, util
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
import torch

print("=" * 60)
print("Sentence Transformers 嵌入模型")
print("=" * 60)

# 设置中文
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 1. 加载模型
print("\n1. 加载预训练模型")

# 常用的Sentence Transformer模型
models_to_try = [
    "all-MiniLM-L6-v2",   # 轻量级，384维
    "all-mpent-base-v2",  # 高质量，768维
]

# 使用轻量级模型
model_name = "all-MiniLM-L6-v2"
model = SentenceTransformer(model_name)

print(f"模型：{model_name}")
print(f"向量维度：{model.get_sentence_embedding_dimension()}")
print(f"最大序列长度:{model.max_seq_length}")

sentences = [
    "The weather is beautiful today.",
    "It's raining outside.",
    "I love programming in Python.",
    "Python is my favorite programming language.",
    "Machine learning is fascinating.",
    "Deep learning uses neural networks.",
]

# 获取嵌入向量
embeddings = model.encode(sentences)
print(f"句子数量：{len(sentences)}")
print(f"嵌入矩阵形状：{embeddings.shape}")

# 显示前几个维度的值
print(f"\n第一个句子的前10个维度")
print(f"{embeddings[0][:10]}")

# 3. 计算相似度
print("\n3. 计算句子相似度")

# 计算所有句子之间的相似度
similarities = util.cos_sim(embeddings,embeddings)

print(f"相似度矩阵形状：{similarities.shape}")
print("\n相似度矩阵")
for i in range(len(sentences)):
    row = similarities[i].tolist()
    print(f"句子{i}:{[f'{s:.3f}' for s in row]}")

# 可视化相似度矩阵
plt.figure(figsize=(10, 8))
sns.heatmap(similarities.numpy(), annot=True, fmt='.3f', cmap='coolwarm',
            xticklabels=[f"S{i}" for i in range(len(sentences))],
            yticklabels=[f"S{i}" for i in range(len(sentences))])
plt.title('句子相似度矩阵')
plt.savefig('./similarity_matrix.png', dpi=150)
plt.show()

# 4. 语义搜索
print("\n4. 语义搜索示例")

corpus = [
    "Python is a high-level programming language.",
    "Java is a popular language for enterprise applications.",
    "JavaScript is used for web development.",
    "Machine learning algorithms learn from data.",
    "Deep learning uses artificial neural networks.",
    "Natural language processing helps computers understand text.",
]

# 编码语料库
corpus_embeddings = model.encode(corpus, convert_to_tensor=True)

# 查询
queries = [
    "What is a programming language?",
    "How do computers learn?",
    "Web development tools",
]

for query in queries:
    query_embedding = model.encode(query, convert_to_tensor=True)

    # 计算相似度
    scores = util.cos_sim(query_embedding, corpus_embeddings)[0]

    # 获取Top-3结果
    top_results = torch.topk(scores, k=3)

    print(f"\n查询 {query}")
    for idx,score in zip(top_results.indices,top_results.values):
        print(f"  [{score:.4f}]  {corpus[idx]}")

# 5. 聚类分析
print("\n5. 文本聚类")

from sklearn.cluster import KMeans

# 更多文本
texts = [
    # 编程相关
    "Python is great for data science.",
    "C++ is used for system programming.",
    "JavaScript runs in browsers.",
    "Go is good for concurrent programming.",

    # AI相关
    "Neural networks are powerful models.",
    "Support vector machines are classical ML algorithms.",
    "Decision trees are easy to interpret.",
    "Random forests combine multiple trees.",

    # 天气相关
    "It's sunny today.",
    "The forecast says rain tomorrow.",
    "Winter is coming.",
    "Summer temperatures are rising.",
]

# 编码
text_embeddings = model.encode(texts)

# K-means聚类
k = 3
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
clusters = kmeans.fit_predict(text_embeddings)

# 可视化聚类结果
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
embeddings_2d = pca.fit_transform(text_embeddings)

plt.figure(figsize=(12, 8))
colors = ['red', 'blue', 'green']

for i, (x, y, cluster, text) in enumerate(zip(embeddings_2d[:, 0], embeddings_2d[:, 1], clusters, texts)):
    plt.scatter(x, y, c=colors[cluster], s=100, alpha=0.7)
    plt.annotate(f"{i}", (x, y), fontsize=8, ha='center', va='bottom')

plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('文本聚类可视化（颜色表示聚类）')
plt.savefig('./text_clustering.png', dpi=150)
plt.show()

print("聚类结果:")
for cluster_id in range(k):
    print(f"\n聚类 {cluster_id}:")
    cluster_texts = [texts[i] for i in range(len(texts)) if clusters[i] == cluster_id]
    for text in cluster_texts:
        print(f"  - {text}")

# 6. 不同模型对比
print("\n6. 不同模型对比")

if False:  # 设置为True以运行（需要下载更多模型）
    models = {
        "all-MiniLM-L6-v2": SentenceTransformer("all-MiniLM-L6-v2"),
        "all-mpnet-base-v2": SentenceTransformer("all-mpnet-base-v2"),
    }

    test_sentences = [
        "I love this product!",
        "This is the best purchase ever.",
        "I hate this item.",
        "Terrible quality.",
    ]

    for name, m in models.items():
        emb = m.encode(test_sentences)
        sim = util.cos_sim(emb, emb)
        print(f"\n{name} (维度: {emb.shape[1]}):")
        print(f"  相似度矩阵: {sim[0][1]:.3f}, {sim[2][3]:.3f}")
else:
    print("跳过模型对比（需要下载额外模型）")

# 7. 嵌入质量评估
print("\n7. 嵌入质量评估")

# 计算类内和类间距离
programming_texts = texts[:4]
ai_texts = texts[4:8]
weather_texts = texts[8:12]


def compute_similarity_stats(group1, group2):
    """计算组间平均相似度"""
    emb1 = model.encode(group1)
    emb2 = model.encode(group2)
    similarities = util.cos_sim(emb1, emb2)
    return similarities.mean().item()


# 类内相似度
programming_intra = compute_similarity_stats(programming_texts, programming_texts)
ai_intra = compute_similarity_stats(ai_texts, ai_texts)
weather_intra = compute_similarity_stats(weather_texts, weather_texts)

# 类间相似度
programming_ai = compute_similarity_stats(programming_texts, ai_texts)
programming_weather = compute_similarity_stats(programming_texts, weather_texts)
ai_weather = compute_similarity_stats(ai_texts, weather_texts)

print("嵌入质量评估:")
print(f"  编程类内相似度: {programming_intra:.3f}")
print(f"  AI类内相似度: {ai_intra:.3f}")
print(f"  天气类内相似度: {weather_intra:.3f}")
print(f"  编程 vs AI: {programming_ai:.3f}")
print(f"  编程 vs 天气: {programming_weather:.3f}")
print(f"  AI vs 天气: {ai_weather:.3f}")
print(f"\n好的嵌入应该使类内相似度 > 类间相似度")

# 8. 保存模型和嵌入
print("\n8. 保存模型和嵌入")

# 保存模型
model.save('./sentence_model')
print(f"模型已保存: learn11/sentence_model")

# 保存嵌入
np.save('./sentence_embeddings.npy', embeddings)
print(f"嵌入已保存: learn11/sentence_embeddings.npy")

print("\n✅ Sentence Transformers演示完成")
