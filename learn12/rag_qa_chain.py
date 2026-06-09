from langchain_community.vectorstores import Chroma    #存储向量、相似度搜、MMR检索、Retriever
from langchain_community.embeddings import HuggingFaceEmbeddings  #文本->向量
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA, ConversationalRetrievalChain  #最经典RAG链,带记忆RAG
from langchain.memory import ConversationBufferMemory                   #ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
import matplotlib.pyplot as plt
import time

print("=" * 60)
print("完整RAG问答链")
print("=" * 60)

# 1. 准备知识库文档
print("\n1. 准备知识库文档")

knowledge_base = """
# RAG（检索增强生成）技术详解

## 什么是RAG？
RAG（Retrieval-Augmented Generation）是一种结合信息检索和文本生成的技术架构。
它通过先从知识库中检索相关信息，再将检索结果作为上下文提供给LLM生成答案。

## RAG的核心优势
1. **知识实时更新**：只需更新知识库，无需重新训练模型
2. **减少幻觉**：基于检索到的事实信息生成答案
3. **可追溯性**：可以引用信息来源
4. **成本低廉**：相比微调，RAG成本更低

## RAG的工作流程
- 第一步：用户提出问题
- 第二步：将问题向量化
- 第三步：在向量数据库中进行相似度检索
- 第四步：将检索到的相关内容作为上下文
- 第五步：LLM基于上下文生成答案

## 实现RAG的关键组件
- **文档加载器**：加载各种格式的文档
- **文本分割器**：将长文档切分成合适的块
- **嵌入模型**：将文本转换为向量
- **向量数据库**：存储和检索向量
- **大语言模型**：生成最终答案

## LangChain中的RAG实现
LangChain提供了完整的RAG工具链：
- Document Loaders：加载文档
- Text Splitters：文本分割
- Embeddings：向量化
- Vector Stores：向量存储
- RetrievalQA：检索问答链
- ConversationalRetrievalChain：对话式检索链
"""

# 创建文档
with open('./knowledge_base.txt', 'w', encoding='utf-8') as f:
    f.write(knowledge_base)

# 2. 加载和分割文档
print("\n2. 加载和分割文档")

loader = TextLoader('./knowledge_base.txt', encoding='utf-8')
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50,
    separators=["\n\n", "\n", "。", "；", "，", " ", ""]
)
chunks = text_splitter.split_documents(documents)

print(f"原始文档: 1个")
print(f"分割后块数: {len(chunks)}")

# 3. 创建向量存储
print("\n3. 创建向量存储")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    model_kwargs={'device': 'cuda'}
)

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./rag_db"
)

print(f"向量数据库创建完成，包含 {len(chunks)} 个向量")

# 4. 使用Mock LLM（实际使用时替换为真实模型）
print("\n4. 初始化LLM")


# 由于本地运行LLM需要较大资源，这里使用一个模拟的响应函数
# 实际使用时可以替换为：ChatOpenAI、HuggingFacePipeline等

class MockLLM:
    """模拟LLM响应（用于演示RAG流程）"""

    def __call__(self, prompt):
        return self._generate_response(prompt)

    def _generate_response(self, prompt):
        # 模拟基于上下文的回答
        if "RAG" in prompt:
            return "RAG（检索增强生成）是一种结合信息检索和文本生成的技术架构。它通过先从知识库中检索相关信息，再将检索结果作为上下文提供给LLM生成答案。"
        elif "向量数据库" in prompt:
            return "向量数据库是专门用于存储和检索高维向量的数据库，是实现RAG系统的关键组件。常见的向量数据库包括Chroma、FAISS、Milvus等。"
        elif "LangChain" in prompt:
            return "LangChain是一个用于构建LLM应用的开发框架，提供了Document Loaders、Text Splitters、Vector Stores、RetrievalQA等完整的RAG工具链。"
        else:
            return "根据知识库信息，我无法完全回答这个问题。请提供更多上下文或询问其他问题。"


mock_llm = MockLLM()
print("使用模拟LLM（实际使用时请替换为真实模型）")

# 5. 创建Prompt模板
print("\n5. 创建Prompt模板")

prompt_template = """你是一个专业的技术助手。请基于以下上下文信息回答用户的问题。

上下文信息:
{context}

用户问题: {question}

回答要求:
1. 严格基于上下文信息回答
2. 如果上下文中没有相关信息，请如实告知
3. 回答要准确、简洁、专业

回答:"""

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]   #上下文和问题
)

# 6. 创建检索QA链
print("\n6. 创建检索QA链")

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

qa_chain = RetrievalQA.from_chain_type(
    llm=mock_llm,
    chain_type="stuff",   #检索出来的多个文档,如何交给LLM,此处全部文档直接塞进去
    retriever=retriever,
    chain_type_kwargs={"prompt": PROMPT},   #指定Prompt模板
    return_source_documents=True            #是否返回检索文档
)
"""
chain_type="stuff",   #检索出来的多个文档,如何交给LLM,此处全部文档直接塞进去
map_reduce:先分别总结,再汇总总结,适合超长文档
refine：渐进式推理
map_rerank：每个文档单独回答，打分，选最高分
"""

print("检索QA链创建成功")

# 7. 测试问答
print("\n7. 测试问答")

test_questions = [
    "什么是RAG？",
    "RAG有哪些优势？",
    "向量数据库的作用是什么？",
    "LangChain如何实现RAG？",
    "RAG和微调有什么区别？"
]

print("\n问答测试结果:")
for question in test_questions:
    print(f"\n问: {question}")

    start_time = time.time()
    result = qa_chain.invoke({"query": question})
    elapsed_time = time.time() - start_time

    print(f"答: {result['result']}")
    print(f"检索文档数: {len(result['source_documents'])}")
    print(f"耗时: {elapsed_time:.2f}秒")

# 8. 对话式检索链（带记忆）
print("\n8. 对话式检索链")

memory = ConversationBufferMemory(
    memory_key="chat_history",   #历史记录保存到
    return_messages=True,        #保存Message对象
    output_key="answer"          #AI输出字段名称
)

conversational_chain = ConversationalRetrievalChain.from_llm(
    llm=mock_llm,
    retriever=retriever,
    memory=memory,
    return_source_documents=True
)

print("对话式检索链创建成功")

# 多轮对话测试
print("\n多轮对话测试:")
conversations = [
    "什么是RAG？",
    "它有哪些优势？",  # 这里的"它"应该能理解指代RAG
    "那LangChain是什么？",
]

for question in conversations:
    print(f"\n问: {question}")
    result = conversational_chain.invoke({"question": question})
    print(f"答: {result['answer']}")

# 9. 检索效果分析
print("\n9. 检索效果分析")


def analyze_retrieval_quality(chain, questions):
    """分析检索质量"""
    results = []

    for question in questions:
        result = chain.invoke({"query": question})
        sources = [doc.metadata.get('source', 'unknown') for doc in result['source_documents']]

        results.append({
            'question': question,
            'num_sources': len(result['source_documents']),
            'sources': sources,
            'answer_length': len(result['result'])
        })

    return results


analysis = analyze_retrieval_quality(qa_chain, test_questions)

# 可视化检索分析
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

questions_short = [q[:15] + "..." if len(q) > 15 else q for q in test_questions]
source_counts = [r['num_sources'] for r in analysis]
answer_lengths = [r['answer_length'] for r in analysis]

axes[0].bar(range(len(questions_short)), source_counts)
axes[0].set_xticks(range(len(questions_short)))
axes[0].set_xticklabels(questions_short, rotation=45, ha='right')
axes[0].set_ylabel('检索文档数')
axes[0].set_title('各问题的检索文档数')

axes[1].bar(range(len(questions_short)), answer_lengths)
axes[1].set_xticks(range(len(questions_short)))
axes[1].set_xticklabels(questions_short, rotation=45, ha='right')
axes[1].set_ylabel('答案长度 (字符)')
axes[1].set_title('各问题的答案长度')

plt.tight_layout()
plt.savefig('./qa_analysis.png', dpi=150)
plt.show()

# 10. 保存配置
import json

rag_config = {
    'chunk_size': 300,
    'chunk_overlap': 50,
    'embedding_model': "paraphrase-multilingual-MiniLM-L12-v2",
    'retrieval_k': 3,
    'chain_type': "stuff",
    'vectorstore_path': "./rag_db",
    'num_chunks': len(chunks)
}

with open('./rag_config.json', 'w') as f:
    json.dump(rag_config, f, indent=2)

print("\nRAG配置已保存: ./rag_config.json")

# 11. 保存向量数据库
vectorstore.persist()
print("向量数据库已持久化: ./rag_db")

print("\n✅ 完整RAG问答链完成")