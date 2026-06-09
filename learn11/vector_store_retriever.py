from langchain_community.vectorstores import Chroma,FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document  #LangChain中的文档对象(文本内容+元数据)
import matplotlib.pyplot as plt
import numpy as np
import time

print("=" * 60)
print("向量存储与检索")
print("=" * 60)

# 1. 初始化嵌入模型
print("\n1. 初始化嵌入模型")

# 使用轻量级中文嵌入模型
embeddings = HuggingFaceEmbeddings(
    model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    model_kwargs={'device':'cuda'},
    encode_kwargs = {'normalize_embeddings':True}
)
print(f"嵌入模型加载成功")
print(f"向量维度：{len(embeddings.embed_query('测试文本'))}")  #List[float]

# 2. 准备文档
print("\n2. 准备文档")

documents_texts = [
    "Python是一种高级编程语言，以简洁易读著称。它广泛应用于数据科学、Web开发等领域。",
    "机器学习是人工智能的核心领域，使计算机能够从数据中学习。",
    "深度学习使用多层神经网络，在图像识别、自然语言处理方面表现出色。",
    "RAG（检索增强生成）结合了信息检索和文本生成技术。",
    "向量数据库专门用于存储和检索高维向量，是实现RAG的关键组件。",
    "LangChain是一个用于构建LLM应用的开发框架，提供了丰富的组件。",
    "Transformer模型使用注意力机制，成为现代NLP的基础架构。",
    "BERT是预训练语言模型，通过掩码语言建模任务学习文本表示。",
    "GPT系列模型使用自回归方式生成文本，擅长对话和创作任务。",
    "Prompt Engineering是优化LLM输出的重要技术。"
]

# 创建Document对象
documents = [Document(page_content=text,metadata={"id":i,"category":"AI"})
             for i ,text in enumerate(documents_texts)]
print(f"文档数量:{len(documents)}")

# 3. 创建向量存储
print("\n3. 创建向量存储")

# Chroma向量数据库
start_time = time.time()
chroma_vectorstore = Chroma.from_documents(
    documents=documents,
    embedding = embeddings,
    persist_directory='./chroma_db'
)
chroma_time = time.time() - start_time
print(f"Chroma创建完成，耗时：{chroma_time:.2f}秒")

# FAISS向量数据库（备选）
# faiss_vectorstore = FAISS.from_documents(
#     documents=documents,
#     embedding = embeddings
# )
# print(f"FAISS创建完成")

# 4.基础检索
print("\n4. 基础检索")

query = "什么是机器学习？"
retrieved_docs = chroma_vectorstore.similarity_search(query,k=3)

print(f"查询：{query}")
print(f"检索结果:")
for i,doc in enumerate(retrieved_docs):
    print(f" {i+1}. {doc.page_content}")

# 5. 带分数的检索
print("\n5. 带分数的检索")
results_with_scores = chroma_vectorstore.similarity_search_with_score(query,k=3)
print(f"查询: {query}")
print(f"检索结果（带相似度分数）:")
for doc, score in results_with_scores:
    print(f"  {doc.page_content[:50]}...")
    print(f"    分数: {score:.4f} (越小越相似)")

# 6. 最大边际相关性检索（MMR）
print("\n6. 最大边际相关性检索 (MMR)")

mmr_results = chroma_vectorstore.max_marginal_relevance_search(
    query,
    k=3,
    fetch_k=10,
    lambda_mult=0.5  # 多样性参数
)

print(f"MMR检索结果（多样性增强）：")
for doc in mmr_results:
    print(f" {doc.page_content[:60]}...")

# 7.创建Retriever
print(f"\n7.创建Retriever")

retriever = chroma_vectorstore.as_retriever(
    search_kwargs={'k':3}
)
print(f"Retriever配置:")
print(f"  search_type: similarity")
print(f"  search_kwargs: k=3")

# 8. 检索质量评估
print("\n8. 检索质量评估")

test_queries = [
    ("Python编程", [0]),      # 期望返回文档0
    ("机器学习算法", [1, 2]),  # 期望返回文档1、2
    ("RAG技术", [3, 4]),      # 期望返回文档3、4
    ("GPT模型", [8]),         # 期望返回文档8
]


def evaluate_retrieval(vectorstore, test_queries, k=3):
    """评估检索质量"""
    results = []
    for query, expected_indices in test_queries:
        retrieved = vectorstore.similarity_search(query, k=k)
        retrieved_indices = [int(doc.metadata['id']) for doc in retrieved]

        # 计算Recall@k
        expected_set = set(expected_indices)
        retrieved_set = set(retrieved_indices[:len(expected_set)])
        recall = len(expected_set & retrieved_set) / len(expected_set) if expected_set else 1

        results.append({
            'query': query,
            'expected': expected_indices,
            'retrieved': retrieved_indices,
            'recall@k': recall
        })

    return results


evaluation_results = evaluate_retrieval(chroma_vectorstore, test_queries)

print("\n检索质量评估:")
for result in evaluation_results:
    print(f"查询: {result['query']}")
    print(f"  期望文档: {result['expected']}")
    print(f"  检索文档: {result['retrieved']}")
    print(f"  Recall@3: {result['recall@k']:.2f}")




