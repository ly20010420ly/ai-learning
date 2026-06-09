from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,  #按照优先级逐层切分  #递归字符分割器
    CharacterTextSplitter,           #按照指定字符切分   #字符分割器
    TokenTextSplitter,               #按照 Token 数量切分
    MarkdownHeaderTextSplitter       #专门处理 Markdown
)
import matplotlib.pyplot as plt
import numpy as np

# 设置中文
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


print("=" * 60)
print("文本分割器")
print("=" * 60)

# 1. 准备测试文本
print("\n1. 准备测试文本")

long_text = """
LangChain是一个用于开发由语言模型驱动的应用程序的框架。

第一章：核心概念

LangChain提供了模块化的组件，包括LLM、Prompt、Chain、Memory、Retriever等。
这些组件可以灵活组合，构建复杂的AI应用。

第二章：链式调用

Chain是LangChain的核心概念，它将多个组件串联成一个完整的处理流程。
例如：LLMChain将PromptTemplate和LLM组合成一个简单的问答链。

第三章：记忆管理

Memory组件负责管理多轮对话的上下文。
ConversationBufferMemory可以存储完整的对话历史。
ConversationSummaryMemory则对历史进行摘要压缩。

第四章：检索增强

Retriever组件负责从知识库中检索相关文档。
VectorStoreRetriever与向量数据库集成，实现语义搜索。

第五章：代理

Agent是LangChain的高级特性，它可以决定使用哪些工具来完成任务。
Agent能够根据用户输入动态选择工具并执行。

LangChain支持的LLM包括：OpenAI GPT系列、Anthropic Claude、Hugging Face模型等。
"""

print(f"原始文本长度: {len(long_text)} 字符")

# 2. 递归字符分割器
print("\n2. 递归字符分割器 (RecursiveCharacterTextSplitter)")

recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,   #每块最大字符数
    chunk_overlap=20,  #块间重叠字符数
    separators=["\n\n","\n","。","；","，"," ",""],
    length_function=len
)

recursive_chunks = recursive_splitter.split_text(long_text)
print(f"分割块数：{len(recursive_chunks)}")
print("分割结果：")
for i,chunk in enumerate(recursive_chunks[:3]):
    print(f" 块:{i+1}:{chunk[:80]}...(长度：{len(chunk)})")

# 3. 字符分割器
print("\n3. 字符分割器 (CharacterTextSplitter)")

char_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=200,
    chunk_overlap=20,
    length_function=len
)

char_chunks = char_splitter.split_text(long_text)
print(f"分割块数：{len(char_chunks)}")
for i,chunk in enumerate(char_chunks[:3]):
    print(f"  块{i+1}：{chunk[:80]}...")

# 4. Token分割器
print("\n4. Token分割器 (TokenTextSplitter)")

token_splitter = TokenTextSplitter(
    chunk_size=200,  #Token数量
    chunk_overlap=10,
    encoding_name="cl100k_base"   #GPT-4使用的编码
)
token_chunks = token_splitter.split_text(long_text)
print(f"分割块数：{len(token_chunks)}")
print(f"第一块token数估计：~{len(token_chunks[0])} 字符")

# 5. Markdown标题分割器
print("\n5. Markdown标题分割器 (MarkdownHeaderTextSplitter)")

markdown_text = """
# 第一章 RAG基础

## 1.1 什么是RAG
RAG是检索增强生成的缩写，结合了信息检索和文本生成。

## 1.2 RAG的优势
- 知识可实时更新
- 减少幻觉
- 可接入私有数据

# 第二章 LangChain实战

## 2.1 核心组件
包括Document Loaders、Text Splitters、Vector Stores等。

## 2.2 构建RAG系统
五步构建：加载→分割→向量化→检索→生成
"""

headers_to_split_on = [
    ("#","Header 1"),
    ("##","Header 2"),
]

markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
md_chunks = markdown_splitter.split_text(markdown_text)

print(f"分割块数：{len(md_chunks)}")
for chunk in md_chunks:
    print(f"  标题：{chunk.metadata},内容:{chunk.page_content[:50]}...")

# 6. 分割策略对比
print("\n6. 分割策略对比")

def evaluate_splitter(splitter,text,name):
    """评估分割器效果"""
    chunks = splitter.split_text(text)

    # 统计信息
    chunk_lengths = [len(c) for c  in chunks]

    return  {
        'name':name,
        'num_chunks':len(chunks),
        'avg_length':np.mean(chunk_lengths) if chunk_lengths else 0,
        'std_length':np.std(chunk_lengths) if chunk_lengths else 0,
        'min_length':np.min(chunk_lengths) if chunk_lengths else 0,
        'max_length':np.max(chunk_lengths) if chunk_lengths else 0
    }

splitters = [
    (recursive_splitter,"Recursive Splitter"),
    (char_splitter,"Character Splitter"),
    (token_splitter,"Token Splitter")
]

results = []
for splitter,name in splitters:
    result = evaluate_splitter(splitter,long_text,name)
    results.append(result)
    print(f"\n{name}")
    print(f"  块数:{result['num_chunks']}")
    print(f"  平均长度：{result['avg_length']:.1f}")
    print(f"  长度标准差：{result['std_length']:.1f}")

# 7. 可视化分割效果
print("\n7. 可视化分割效果")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 块长度分布
for result in results:
    axes[0].bar(result['name'], result['avg_length'],
                yerr=result['std_length'], capsize=5, alpha=0.7)
axes[0].set_ylabel('平均块长度 (字符)')
axes[0].set_title('不同分割器的平均块长度')

# 块数量对比
names = [r['name'] for r in results]
num_chunks = [r['num_chunks'] for r in results]
axes[1].bar(names, num_chunks, alpha=0.7)
axes[1].set_ylabel('块数量')
axes[1].set_title('不同分割器的块数量')

plt.tight_layout()
plt.savefig('./splitter_comparison.png', dpi=150)
plt.show()

# 8. 最佳实践建议
print("\n8. 分割器选择建议")

best_practices = """
┌─────────────────────────────────────────────────────────────────┐
│ 场景                    │ 推荐分割器              │ 参数建议        │
├─────────────────────────────────────────────────────────────────┤
│ 通用文本                │ RecursiveCharacterTextSplitter │ chunk_size=500-1000 │
│ 代码                    │ RecursiveCharacterTextSplitter │ separators=["\\n\\n", "\\n", ";", " ", ""] │
│ Markdown文档            │ MarkdownHeaderTextSplitter │ 保留标题层级   │
│ 多语言文档              │ RecursiveCharacterTextSplitter │ 根据语言调整分隔符 │
│ Token限制严格           │ TokenTextSplitter        │ chunk_size=512 │
└─────────────────────────────────────────────────────────────────┘
"""
print(best_practices)

