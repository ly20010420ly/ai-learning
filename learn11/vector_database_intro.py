import chromadb
from chromadb.utils import embedding_functions

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd

# 设置中文
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("向量数据库入门 - ChromaDB")
print("=" * 60)

# 1. 初始化ChromaDB客户端
print("\n1. 初始化ChromaDB")

# 使用持续化客户端（数据保存在磁盘）
client = chromadb.PersistentClient(path='./chroma_db')

# 创建emnedding函数（使用Sentence Transformers）
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name = "all-MiniLM-L6-v2"
)

print(f"客户端创建成功")
print(f"数据存储路径：./chroma_db")

# 2. 创建集合（Collection）

try:
    client.delete_collection("my_documents")
except:
    pass

# 创建新集合
collection = client.create_collection(
    name="my_documents",
    embedding_function=embedding_fn,
    metadata={'description':'示例文档集合'}
)

print(f"集合创建成功：{collection.name}")
print(f"集合元数据:{collection.metadata}")

# 3. 添加文档
print("\n3. 添加文档")

documents = [
    "Python is a high-level programming language known for its simplicity.",
    "Machine learning is a subset of AI that enables systems to learn from data.",
    "Deep learning uses neural networks with multiple layers.",
    "Natural language processing helps computers understand human language.",
    "Computer vision enables machines to interpret and understand visual information.",
    "Reinforcement learning trains agents through rewards and punishments.",
    "Transformers are neural network architectures that use attention mechanisms.",
    "BERT is a pre-trained transformer model for natural language understanding.",
]

# 添加元数据
metadatas = [
    {"category": "programming", "difficulty": "beginner"},
    {"category": "ai", "difficulty": "intermediate"},
    {"category": "ai", "difficulty": "advanced"},
    {"category": "ai", "difficulty": "intermediate"},
    {"category": "ai", "difficulty": "advanced"},
    {"category": "ai", "difficulty": "advanced"},
    {"category": "ai", "difficulty": "advanced"},
    {"category": "nlp", "difficulty": "advanced"},
]

ids = [f"doc_{i}" for i in range(len(documents))]

# 添加文档
collection.add(
    documents = documents,
    metadatas = metadatas,
    ids=ids
)

print(f"已添加 {len(documents)} 个文档")
print(f"文档ID: {ids}")

# 4.查看集合信息

print(f"集合名称：{collection.name}")
print(f"文档数量：{collection.count()}")
print(f"集合元数据：{collection.metadata}")

# 5. 基本查询
print("\n5. 基本查询 - 相似度搜索")

query = "What is Python?"
result = collection.query(
    query_texts=[query],
    n_results=3
)

print(f"查询;{query}")
print(f"检索结果：")
for i,(doc,distance,metadata,id_) in enumerate(zip(
    result['documents'][0],
    result['distances'][0],
    result['metadatas'][0],
    result['ids'][0]
)):
    print(f"\n {i+1}. ID:{id_}")
    print(f"  文档：{doc[:80]}...")
    print(f"  距离：{distance:.4f}")
    print(f"  元数据：{metadata}")

# 6. 带元数据过滤的查询
print("\n6. 带元数据过滤的查询")

query = "How do neural networks work?"
results = collection.query(
    query_texts = [query],
    n_results=3,
    where={'difficulty':'advanced'}
)

print(f"查询：{query}")
print(f"过滤条件：difficulty = advanced")
print(f"检索结果：")
for i,doc in enumerate(results['documents'][0]):
    print(f" {i+1}. {doc[:80]}...")

# 7. 批量查询
print("\n7. 批量查询")

queries = [
    "What is machine learning?",
    "Tell me about computer vision",
    "Explain reinforcement learning"
]

for query in queries:
    result = collection.query(
        query_texts = [query],
        n_results=2
    )
    print(f"查询:{query}")
    for i,doc in enumerate(result['documents'][0]):
        print(f" 结果{i+1}：{doc[:60]}...")

# 8. 获取和更新文档
print("\n8. 获取和更新文档")

# 获取文档
doc = collection.get(ids="doc_0")
print(f"获取文档:doc_0")
print(f"  内容：{doc['documents'][0]}")
print(f"  元数据:{doc['metadatas'][0]}")

# 更新文档
collection.update(
    ids="doc_0",
    metadatas=[{"category": "programming", "difficulty": "beginner", "updated": True}]
)
updated_doc = collection.get(ids=["doc_0"])
print(f"\n更新后的文档:")
print(f"  元数据: {updated_doc['metadatas'][0]}")

# 9. 删除文档
print("\n9. 删除文档")

# 删除单个文档
collection.delete(ids=["doc_0"])
print(f"删除 doc_0 后，文档数量: {collection.count()}")

# 重新添加回来（用于后续演示）
collection.add(
    documents=[documents[0]],
    metadatas=[{"category": "programming", "difficulty": "beginner"}],
    ids=["doc_0"]
)
print(f"重新添加后，文档数量: {collection.count()}")

# 10. 向量可视化
print("\n10. 向量可视化")

# 获取所有文档的向量
# 注意：ChromaDB默认不直接返回向量，我们需要重新embedding
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
doc_embeddings = model.encode(documents)

# PCA降维到2D
pca = PCA(n_components=2)
embeddings_2d = pca.fit_transform(doc_embeddings)

# 查询向量
query = "What is AI?"
query_embedding = model.encode([query])
query_2d = pca.transform(query_embedding)

# 绘制
plt.figure(figsize=(12, 8))

# 散点图
categories = [m['category'] for m in metadatas]
colors = {'programming': 'blue', 'ai': 'green', 'nlp': 'purple'}

for i, (x, y, category) in enumerate(zip(embeddings_2d[:, 0], embeddings_2d[:, 1], categories)):
    plt.scatter(x, y, c=colors.get(category, 'gray'), s=100, alpha=0.7)
    plt.annotate(f"doc_{i}", (x, y), fontsize=8, ha='center', va='bottom')

# 查询点
plt.scatter(query_2d[0, 0], query_2d[0, 1], c='red', s=200, marker='*', label='Query')

# 添加图例
for category, color in colors.items():
    plt.scatter([], [], c=color, label=category)
plt.legend()

plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('文档向量PCA可视化（颜色表示类别）')
plt.grid(True, alpha=0.3)
plt.savefig('./vector_visualization.png', dpi=150)
plt.show()

# 11. 不同距离度量的效果
print("\n11. 距离度量方式")

print("""
ChromaDB默认使用余弦距离，其他可选:

1. 余弦距离 (cosine): 1 - cosine_similarity
   - 范围: [0, 2]
   - 适合: 文本相似度
   - 特点: 只关心方向，不关心长度

2. L2距离 (欧氏距离): sqrt(sum((x-y)^2))
   - 范围: [0, ∞)
   - 适合: 稠密向量
   - 特点: 考虑向量长度

3. IP距离 (内积): -dot(x, y)
   - 范围: (-∞, ∞)
   - 适合: 归一化向量
   - 特点: 值越大越相似

示例: 同一文档在不同距离下的结果
""")

# 12. 保存集合统计信息
collection_stats = {
    'name': collection.name,
    'count': collection.count(),
    'metadata': collection.metadata,
    'documents_sample': documents[:3],
    'categories': list(set(categories))
}

import json
with open('./collection_stats.json', 'w') as f:
    json.dump(collection_stats, f, indent=2)

print("\n 向量数据库入门完成")
print(f"集合统计已保存: learn11/collection_stats.json")
