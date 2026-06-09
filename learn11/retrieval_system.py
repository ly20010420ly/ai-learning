import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
from typing import List,Dict,Any  #给 Python 代码添加类型提示（Type Hint）
import matplotlib.pyplot as plt
import json

# 设置中文
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("完整检索系统实现")
print("=" * 60)

# 1. 定义检索器类
print("\n1. 构建检索器类")

class DocumentRetriever:
    """文件检索系统"""
    def __init__(self,collection_name="knowledge_base",persist_dir="./retriever_db"):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name = "all-MiniLM-L6-v2"
        )
        self.collection_name = collection_name
        self.collection = None
        self.__init__collection()

    def __init__collection(self):
        """初始化集合"""
        try:
            # 尝试获取已有集合
            self.collection = self.client.get_collection(
                name = self.collection_name,
                embedding_function=self.embedding_fn
            )
            print(f"加载已有集合;{self.collection_name}")
        except:
            # 创建新集合
            self.collection = self.client.create_collection(
                name = self.collection_name,
                embedding_function=self.embedding_fn,
                metadata={"description":"知识库"}
            )
            print(f"创建新集合：{self.collection_name}")

    def add_documents(self,documents:List[str],metadatas:List[Dict] = None,ids:List[str]= None):
        """添加文档"""
        if ids is None:
            ids = [f"doc_{i}" for i in range(len(documents))]

        if metadatas is None:
            metadatas = [
                {"source": "default"}
                for _ in documents
            ]

        self.collection.add(
            documents = documents,
            metadatas = metadatas,
            ids = ids
        )
        print(f"添加{len(documents)}个文档")

    def search(self,query:str,top_k:int=5,filter_criteria:Dict=None)->List[Dict]:
        """搜索相关文档"""
        results = self.collection.query(
            query_texts = [query],
            n_results = top_k,
            where = filter_criteria
        )
        # 格式化结果
        formatted_results = []
        for i in range(len(results['documents'][0])):
            formatted_results.append({
                'id':results['ids'][0][i],
                'document':results['documents'][0][i],
                'distance':results['distances'][0][i],  #可能性
                'metadata':results['metadatas'][0][i]
            })
        return formatted_results

    def get_stats(self)->Dict:
        """获取统计信息"""
        return {
            'name':self.collection.name,
            'count':self.collection.count(),
            'metadata':self.collection.metadata
        }

    def delete_document(self,doc_id:str):
        """删除文档"""
        self.collection.delete(ids=[doc_id])
        print(f"删除文档：{doc_id}")

# 2. 创建示例知识库
print("\n2. 创建示例知识库")
retriever = DocumentRetriever(collection_name="knowledge_base")

# 准备文档
documents = [
    # Python相关
    "Python是一种解释型、面向对象的高级编程语言。它具有简洁的语法和强大的标准库。",
    "Python的装饰器是一种函数，用于修改其他函数的行为，是Python的高级特性之一。",
    "Python中的列表推导式提供了一种简洁的创建列表的方法。",

    # 机器学习相关
    "机器学习是人工智能的一个子集，它使系统能够从数据中自动学习和改进。",
    "监督学习使用带标签的数据训练模型，包括分类和回归任务。",
    "深度学习使用多层神经网络，在图像识别、自然语言处理等领域取得了突破性进展。",

    # RAG相关
    "RAG（检索增强生成）结合了信息检索和大语言模型，能够生成更准确、更有依据的回答。",
    "向量数据库专门用于存储和检索高维向量，是实现RAG系统的关键组件。",
    "嵌入模型将文本转换为向量表示，使得语义相似的文本在向量空间中靠近。",

    # 面试相关
    "机器学习面试常问的问题包括：过拟合与欠拟合、偏差与方差、正则化方法等。",
    "Python面试中经常考察装饰器、生成器、上下文管理器等高级特性。",
    "RAG系统面试会问：如何优化检索质量、如何处理长文档、如何评估系统性能等。",
]

# 元数据
metadatas = [
    {"category": "python", "topic": "基础", "difficulty": "easy"},
    {"category": "python", "topic": "高级", "difficulty": "hard"},
    {"category": "python", "topic": "语法", "difficulty": "medium"},
    {"category": "ml", "topic": "基础", "difficulty": "easy"},
    {"category": "ml", "topic": "基础", "difficulty": "medium"},
    {"category": "ml", "topic": "深度学习", "difficulty": "hard"},
    {"category": "rag", "topic": "基础", "difficulty": "medium"},
    {"category": "rag", "topic": "技术", "difficulty": "hard"},
    {"category": "rag", "topic": "技术", "difficulty": "medium"},
    {"category": "interview", "topic": "ml", "difficulty": "medium"},
    {"category": "interview", "topic": "python", "difficulty": "medium"},
    {"category": "interview", "topic": "rag", "difficulty": "hard"},
]

ids = [f"doc_{i}" for i in range(len(documents))]

# 添加文档
retriever.add_documents(documents, metadatas, ids)

print(f"知识库统计: {retriever.get_stats()}")

# 3. 搜索演示
print("\n3. 搜索演示")

queries = [
    "什么是Python？",
    "机器学习如何工作？",
    "RAG系统是什么？",
    "面试会问什么Python问题？",
]

for query in queries:
    print(f"\n查询: {query}")
    results = retriever.search(query, top_k=2)
    for i, result in enumerate(results):
        print(f"  {i+1}. [{result['distance']:.4f}] {result['document'][:80]}...")
        print(f"     类别: {result['metadata'].get('category', 'N/A')}")

# 4. 带过滤的搜索
print("\n4. 带过滤的搜索")

query = "深度学习"
print(f"查询: {query}")

# 不过滤
results_no_filter = retriever.search(query, top_k=3)
print("\n不过滤:")
for r in results_no_filter:
    print(f"  - {r['document'][:60]}... (类别: {r['metadata'].get('category', 'N/A')})")

# 只检索机器学习相关
results_filtered = retriever.search(query, top_k=3, filter_criteria={"category": "ml"})
print("\n只检索ML类别:")
for r in results_filtered:
    print(f"  - {r['document'][:60]}... (类别: {r['metadata'].get('category', 'N/A')})")

# 5. 创建多个知识库
print("\n5. 创建多个知识库")

# 创建不同领域的知识库
domains = {
    "python_qa": [
        "Q: Python中的GIL是什么？ A: GIL是全局解释器锁，限制了多线程的并行执行。",
        "Q: 什么是装饰器？ A: 装饰器是一个函数，用于修改其他函数的行为。",
        "Q: 列表和元组有什么区别？ A: 列表可变，元组不可变。",
    ],
    "ml_qa": [
        "Q: 什么是过拟合？ A: 模型在训练集上表现太好，但在测试集上表现差。",
        "Q: 什么是交叉验证？ A: 将数据分成多份，轮流作为验证集。",
        "Q: 什么是梯度下降？ A: 通过计算梯度来最小化损失函数的优化算法。",
    ],
}

for domain_name, domain_docs in domains.items():
    domain_retriever = DocumentRetriever(collection_name=domain_name)
    domain_retriever.add_documents(domain_docs)
    print(f"创建知识库: {domain_name}，包含 {len(domain_docs)} 个文档")

# 6. 检索质量评估
print("\n6. 检索质量评估")


def evaluate_retrieval(retriever, test_queries, relevant_docs):
    """评估检索质量"""
    results = []
    for query, relevant in zip(test_queries, relevant_docs):
        retrieved = retriever.search(query, top_k=3)
        retrieved_ids = [r['id'] for r in retrieved]

        # 计算Recall@3
        relevant_set = set(relevant)
        retrieved_set = set(retrieved_ids)
        recall = len(relevant_set & retrieved_set) / len(relevant_set) if relevant_set else 0

        results.append({
            'query': query,
            'recall@3': recall,
            'retrieved': retrieved_ids,
            'relevant': relevant
        })

    return results


# 定义测试查询和期望的相关文档
test_queries = [
    "Python装饰器",
    "机器学习过拟合",
    "RAG系统检索",
]

relevant_docs = [
    ["doc_1"],  # 文档1是关于装饰器的
    ["doc_3", "doc_4"],  # 文档3是机器学习基础
    ["doc_6", "doc_7"],  # 文档6是RAG基础，文档7是向量数据库
]

evaluation = evaluate_retrieval(retriever, test_queries, relevant_docs)

print("\n检索质量评估结果:")
for result in evaluation:
    print(f"查询: {result['query']}")
    print(f"  Recall@3: {result['recall@3']:.2f}")

# 7. 可视化检索结果
print("\n7. 可视化检索结果")


def visualize_retrieval(query, results):
    """可视化检索结果的距离"""
    distances = [r['distance'] for r in results]
    doc_names = [f"Doc {i + 1}" for i in range(len(results))]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(doc_names, distances, color='skyblue')
    plt.xlabel('Documents')
    plt.ylabel('Distance (smaller = more similar)')
    plt.title(f'检索结果: "{query}"')

    # 添加数值标签
    for bar, dist in zip(bars, distances):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f'{dist:.4f}', ha='center', va='bottom')

    plt.tight_layout()
    return plt.gcf()


# 选择一个查询进行可视化
sample_query = "什么是机器学习？"
sample_results = retriever.search(sample_query, top_k=5)
fig = visualize_retrieval(sample_query, sample_results)
fig.savefig('./retrieval_visualization.png', dpi=150)
plt.show()

# 8. 保存检索系统
print("\n8. 保存检索系统")

# 保存检索器配置
config = {
    'collection_name': retriever.collection_name,
    'persist_dir': '/retriever_db',
    'embedding_model': 'all-MiniLM-L6-v2',
    'num_documents': retriever.get_stats()['count']
}

with open('./retriever_config.json', 'w') as f:
    json.dump(config, f, indent=2)

print(f"检索器配置已保存: learn11/retriever_config.json")

# 导出文档列表
documents_df = pd.DataFrame({
    'id': ids,
    'document': documents,
    'category': [m['category'] for m in metadatas],
    'topic': [m['topic'] for m in metadatas],
    'difficulty': [m['difficulty'] for m in metadatas]
})
documents_df.to_csv('./documents_export.csv', index=False)
print(f"文档已导出: learn11/documents_export.csv")

print("\n 完整检索系统实现完成")


