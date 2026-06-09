import os
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    CSVLoader,
    UnstructuredMarkdownLoader,
    DirectoryLoader
)
from langchain_core.documents import Document
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

print("=" * 60)
print("LangChain 文档加载器")
print("=" * 60)

# 1. 文本文件加载
print("\n1. 文本文件加载")

# 创建示例文本文件
sample_text = """
Langchain是一个用于开发由语言模型驱动的应用程序的框架

他提供了以下核心功能：
1.组件化：提供模块化的组件（LLM、Prompt、Chain等）
2.链式调用：将多个组件组合成完整的处理流程
3.记忆管理：支持多轮对话的上下文记忆
4.与向量数据库集成实现RAG

Langchian支持的LLM包括：OPenAI、Anthropic、Hugging、Face等。
"""
with open('./sample.txt','w',encoding='utf-8') as f:
    f.write(sample_text)

# 加载文本文件
text_loader = TextLoader('./sample.txt',encoding='utf-8')
text_docs = text_loader.load()

print(f"加载文档数量;{len(text_docs)}")
print(f"文档内容预览：{text_docs[0].page_content[:200]}...")
print(f"元数据：{text_docs[0].metadata}")

# 2. PDF文件加载
print("\n2. PDF文件加载")

# 创建一个简单的PDF示例（使用文本模拟，实际需要真实PDF）
# 这里创建一个小型PDF用于演示
def create_sample_pdf():
    """创建示例PDF文件"""
    c = canvas.Canvas("./sample.pdf", pagesize=letter)
    c.drawString(100, 750, "LangChain PDF文档示例")
    c.drawString(100, 720, "本文档用于演示PDF加载器功能")
    c.drawString(100, 690, "内容包括：")
    c.drawString(100, 660, "1. LangChain框架介绍")
    c.drawString(100, 630, "2. 文档加载器使用方法")
    c.drawString(100, 600, "3. RAG系统构建步骤")
    c.save()

create_sample_pdf()
print("示例PDF已创建: learn12/sample.pdf")

# 加载PDF
pdf_loader = PyPDFLoader('./sample.pdf')
pdf_docs = pdf_loader.load()
for i,doc in enumerate(pdf_docs):
    print(f"  第{i}页：{doc.page_content[:100]}...")

# 3. CSV文件加载
print("\n3. CSV文件加载")

csv_content = """name,category,description
Python,编程语言,一种高级编程语言，以简洁著称
LangChain,框架,用于构建LLM应用的开发框架
RAG,技术,检索增强生成技术
Vector Database,数据库,用于存储和检索向量
"""

with open('./sample.csv', 'w', encoding='utf-8') as f:
    f.write(csv_content)

csv_loader = CSVLoader('./sample.csv', encoding='utf-8')
csv_docs = csv_loader.load()

print(f"CSV记录数: {len(csv_docs)}")
for doc in csv_docs[:3]:
    print(f"  {doc.page_content}")

# 4. Markdown文件加载
print("\n4. Markdown文件加载")

markdown_content = """
# RAG系统架构

## 核心组件
- **Retriever**: 负责从知识库检索相关文档
- **Augmenter**: 将检索结果与用户问题组合
- **Generator**: 基于增强后的输入生成回答

## 工作流程
1. 用户输入问题
2. 向量检索相关文档
3. 构建增强Prompt
4. LLM生成答案
"""

with open('./sample.md', 'w', encoding='utf-8') as f:
    f.write(markdown_content)

md_loader = UnstructuredMarkdownLoader('./sample.md')
md_docs = md_loader.load()

print(f"Markdown内容: {md_docs[0].page_content[:200]}...")

# 5. 目录批量加载
print("\n5. 目录批量加载")

# 创建多个文档
documents_data = [
    ("doc1.txt", "Python是一种解释型、面向对象的编程语言。"),
    ("doc2.txt", "机器学习是人工智能的一个子领域。"),
    ("doc3.txt", "深度学习使用多层神经网络进行学习。"),
]

for filename, content in documents_data:
    with open(f'./{filename}', 'w', encoding='utf-8') as f:
        f.write(content)

# 批量加载
dir_loader = DirectoryLoader(
    './',
    glob="*.txt",
    loader_cls=TextLoader,
    loader_kwargs={'encoding': 'utf-8'}
)
dir_docs = dir_loader.load()

print(f"批量加载文档数: {len(dir_docs)}")
for doc in dir_docs:
    print(f"  {doc.metadata['source']}: {doc.page_content[:50]}...")

# 6. 文档元数据处理
print("\n6. 文档元数据处理")

def add_metadata(docs, source_type):
    """为文档添加自定义元数据"""
    for doc in docs:
        doc.metadata['source_type'] = source_type
        doc.metadata['processed_date'] = '2026-06-01'
    return docs

text_docs = add_metadata(text_docs, 'text')
print(f"添加元数据后的文档:")
print(f"  source_type: {text_docs[0].metadata['source_type']}")
print(f"  processed_date: {text_docs[0].metadata['processed_date']}")

# 7. 加载器对比分析
print("\n7. 加载器对比分析")

loader_comparison = {
    'TextLoader': {'适用格式': '.txt', '特点': '简单文本', '使用难度': '低'},
    'PyPDFLoader': {'适用格式': '.pdf', '特点': '保留结构', '使用难度': '中'},
    'CSVLoader': {'适用格式': '.csv', '特点': '表格数据', '使用难度': '低'},
    'DirectoryLoader': {'适用格式': '批量', '特点': '批量处理', '使用难度': '低'},
    'UnstructuredMarkdownLoader': {'适用格式': '.md', '特点': '保留标题', '使用难度': '中'}
}

print("加载器对比:")
for loader, info in loader_comparison.items():
    print(f"  {loader}: {info}")

# 8. 保存加载结果
import json

loader_stats = {
    'total_documents': len(text_docs) + len(pdf_docs) + len(csv_docs) + len(md_docs),
    'sources': {
        'text': len(text_docs),
        'pdf': len(pdf_docs),
        'csv': len(csv_docs),
        'markdown': len(md_docs)
    }
}

with open('./loader_stats.json', 'w') as f:
    json.dump(loader_stats, f, indent=2)

print(f"\n加载统计: {loader_stats}")
print("\n LangChain文档加载器完成")
