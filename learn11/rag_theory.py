import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import numpy as np

print("=" * 60)
print("RAG（检索增强生成）理论基础")
print("=" * 60)

# 设置中文
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 1. RAG简介
print("\n1. 什么是RAG？")

print("""
RAG (Retrieval-Augmented Generation) = 检索 + 生成

核心思想：
  在LLM生成回答之前，先从知识库中检索相关信息，
  然后将检索到的信息作为上下文提供给LLM，生成更准确的答案。

为什么需要RAG？
  ┌─────────────────────────────────────────────────────────┐
  │ 问题                     │ 解决方案                        │
  ├─────────────────────────────────────────────────────────┤
  │ LLM知识有截止日期        │ RAG可接入实时知识库              │
  │ LLM会产生幻觉           │ RAG提供事实依据                  │
  │ LLM不知道私有数据       │ RAG可检索企业内部文档            │
  │ 大模型微调成本高         │ RAG无需训练，即插即用            │
  └─────────────────────────────────────────────────────────┘
""")

# 2. RAG工作流程可视化
print("\n2. RAG工作流程")


def draw_rag_workflow():
    """绘制RAG工作流程图"""
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis('off')

    # 定义步骤和位置
    steps = [
        (1, 4.5, "1. 用户提问\n\"什么是RAG？\""),
        (3, 4.5, "2. 查询向量化\n用Embedding模型"),
        (5, 4.5, "3. 向量检索\n在知识库中搜索"),
        (7, 4.5, "4. 召回相关文档\nTop-K相关内容"),
        (9, 4.5, "5. LLM生成\n结合上下文回答")
    ]

    # 绘制步骤框
    for x, y, text in steps:
        box = FancyBboxPatch((x - 0.8, y - 0.8), 1.6, 1.6,
                             boxstyle="round,pad=0.1",
                             facecolor='lightblue', edgecolor='blue', linewidth=2)
        ax.add_patch(box)
        ax.text(x, y, text, ha='center', va='center', fontsize=9, wrap=True)

    # 绘制箭头
    for i in range(len(steps) - 1):
        ax.annotate('', xy=(steps[i + 1][0] - 0.8, steps[i][1]),
                    xytext=(steps[i][0] + 0.8, steps[i][1]),
                    arrowprops=dict(arrowstyle='->', color='red', lw=2))

    # 知识库示意图
    kb_box = FancyBboxPatch((4, 1), 2, 1.5,
                            boxstyle="round,pad=0.1",
                            facecolor='lightgreen', edgecolor='green', linewidth=2)
    ax.add_patch(kb_box)
    ax.text(5, 1.75, "知识库\n(向量数据库)", ha='center', va='center', fontsize=10)

    # 连接线
    ax.annotate('', xy=(5, 3.7), xytext=(5, 2.5),
                arrowprops=dict(arrowstyle='->', color='green', lw=2))

    ax.set_title("RAG (Retrieval-Augmented Generation) 工作流程", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('./rag_workflow.png', dpi=150)
    plt.show()


draw_rag_workflow()

# 3. RAG的三个核心组件
print("\n3. RAG的三个核心组件")

components = {
    "检索器 (Retriever)": {
        "功能": "从知识库中检索相关文档",
        "技术": "向量检索 (Embedding + 相似度计算)",
        "常用工具": "ChromaDB, FAISS, Pinecone, Weaviate"
    },
    "增强器 (Augmenter)": {
        "功能": "将检索结果与原始问题组合",
        "技术": "Prompt Engineering",
        "常用模板": "Context: {context}\nQuestion: {question}\nAnswer:"
    },
    "生成器 (Generator)": {
        "功能": "基于增强后的输入生成答案",
        "技术": "LLM (GPT, LLaMA, ChatGLM等)",
        "常用模型": "GPT-4, LLaMA-2, Claude, ChatGLM"
    }
}

for component, details in components.items():
    print(f"\n{component}:")
    for key, value in details.items():
        print(f"  {key}: {value}")

# 4. RAG vs 传统方法对比
print("\n4. RAG vs 传统方法对比")

comparison_data = {
    '方法': ['传统LLM', 'Fine-tuning', 'RAG'],
    '知识更新': ['❌ 需要重新训练', '❌ 需要重新训练', '✅ 实时更新'],
    '私有数据': ['❌ 无法获取', '✅ 可以学习', '✅ 可检索'],
    '幻觉问题': ['❌ 严重', '🟡 部分缓解', '✅ 明显改善'],
    '计算成本': ['低', '高', '中'],
    '可解释性': ['低', '中', '高'],
}

fig, ax = plt.subplots(figsize=(12, 4))
ax.axis('tight')
ax.axis('off')
table = ax.table(cellText=[list(comparison_data[k]) for k in comparison_data.keys()],
                 colLabels=list(comparison_data.keys()),
                 cellLoc='center',
                 loc='center',
                 colWidths=[0.12, 0.18, 0.18, 0.18, 0.15, 0.12])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.5)
plt.title("RAG vs 传统方法对比", fontsize=14, fontweight='bold', y=1.1)
plt.savefig('./rag_comparison.png', dpi=150)
plt.show()

# 5. RAG的变体
print("\n5. RAG的常见变体")

variants = """
┌─────────────────────────────────────────────────────────────────┐
│ 变体名称              │ 特点                                     │
├─────────────────────────────────────────────────────────────────┤
│ Naive RAG            │ 检索 → 增强 → 生成，最简单实现            │
│ Advanced RAG         │ 检索前优化（查询重写）+ 检索后重排        │
│ Modular RAG          │ 可插拔模块，灵活组合                      │
│ Self-RAG             │ 模型自我评估检索质量                      │
│ Corrective RAG       │ 检索结果不理想时，主动搜索纠正            │
│ Adaptive RAG         │ 动态决定是否需要检索                      │
└─────────────────────────────────────────────────────────────────┘
"""
print(variants)

# 6. RAG评估指标
print("\n6. RAG评估指标")

metrics = {
    "检索质量指标": {
        "Hit Rate": "检索结果是否包含正确答案",
        "MRR (Mean Reciprocal Rank)": "第一个正确答案的排名倒数",
        "Recall@K": "前K个结果中包含正确答案的比例",
        "NDCG@K": "考虑排序位置的相关性评分"
    },
    "生成质量指标": {
        " faithfulness (忠实度)": "生成内容是否基于检索到的信息",
        "answer relevance": "答案与问题的相关程度",
        "context relevance": "检索到的上下文的相关性",
        "BLEU/ROUGE": "与参考答案的相似度"
    }
}

for category, metric_dict in metrics.items():
    print(f"\n{category}:")
    for metric, desc in metric_dict.items():
        print(f"  {metric}: {desc}")

# 7. RAG应用场景
print("\n7. RAG应用场景")

applications = """
┌─────────────────────────────────────────────────────────────────┐
│ 场景                    │ 示例                                   │
├─────────────────────────────────────────────────────────────────┤
│ 企业知识库问答          │ 员工手册、内部文档、代码库问答          │
│ 客服机器人              │ 产品手册、常见问题解答                  │
│ 法律/医疗咨询           │ 法律法规、医学文献检索                  │
│ 教育辅导                │ 教材内容、习题解答                      │
│ 科研助手                │ 论文检索、文献综述                      │
│ 个人助手                │ 个人笔记、收藏夹管理                    │
└─────────────────────────────────────────────────────────────────┘
"""
print(applications)

# 8. RAG的优势和挑战
print("\n8. RAG的优势与挑战")

print("""
优势:
  ✓ 知识可实时更新，无需重新训练
  ✓ 可接入私有数据，保护数据安全
  ✓ 减少幻觉，提供可追溯的引用来源
  ✓ 成本低，比微调更经济
  ✓ 可解释性强，能看到检索来源

挑战:
  ✗ 检索质量直接影响生成质量
  ✗ 长上下文处理效率问题
  ✗ 检索与生成的协同优化难度
  ✗ 多跳推理能力有限
  ✗ 存在安全隐私风险
""")

# 9. 保存RAG理论笔记
import json

rag_notes = {
    'definition': 'RAG = Retrieval + Augmentation + Generation',
    'components': ['Retriever', 'Augmenter', 'Generator'],
    'applications': ['知识库问答', '客服机器人', '法律咨询', '教育辅导'],
    'advantages': ['实时更新', '私有数据', '减少幻觉', '低成本', '可解释'],
    'challenges': ['检索质量', '长上下文', '协同优化', '多跳推理']
}

with open('./rag_theory_notes.json', 'w') as f:
    json.dump(rag_notes, f, indent=2)

print("\n✅ RAG理论基础完成，笔记已保存")