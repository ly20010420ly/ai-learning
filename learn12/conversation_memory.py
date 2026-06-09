from langchain.memory import (
    ConversationBufferMemory,
    ConversationBufferWindowMemory,
    ConversationSummaryMemory,
    ConversationSummaryBufferMemory
)
from langchain.schema import HumanMessage, AIMessage
import matplotlib.pyplot as plt

print("=" * 60)
print("对话记忆与多轮对话")
print("=" * 60)

# 设置中文
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 1. 不同类型的对话记忆
print("\n1. 对话记忆类型介绍")

memory_types = {
    "ConversationBufferMemory": "存储完整对话历史，适合短对话",
    "ConversationBufferWindowMemory": "只保留最近K轮对话，适合长对话",
    "ConversationSummaryMemory": "定期总结对话摘要，节省空间",
    "ConversationSummaryBufferMemory": "结合缓冲区和摘要，平衡效果与效率"
}

for name, desc in memory_types.items():
    print(f"  {name}: {desc}")

# 2. Buffer Memory（完整存储）
print("\n2. Buffer Memory - 完整对话历史")

buffer_memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# 添加对话
conversations = [
    ("用户", "什么是RAG？"),
    ("助手", "RAG是检索增强生成技术，结合了信息检索和文本生成。"),
    ("用户", "它有什么优势？"),
    ("助手", "RAG的优势包括知识实时更新、减少幻觉、可追溯来源。"),
    ("用户", "如何实现RAG？"),
    ("助手", "实现RAG需要文档加载、文本分割、向量化、检索和生成五个步骤。")
]

for role, message in conversations:
    if role == "用户":
        buffer_memory.chat_memory.add_user_message(message)
    else:
        buffer_memory.chat_memory.add_ai_message(message)

# 查看存储的对话
stored_messages = buffer_memory.chat_memory.messages
print(f"存储的消息数: {len(stored_messages)}")
for msg in stored_messages:
    print(f"  {msg.type}: {msg.content[:50]}...")

# 3. Window Memory（滑动窗口）
print("\n3. Window Memory - 滑动窗口（保留最近3轮）")

window_memory = ConversationBufferWindowMemory(
    memory_key="chat_history",
    k=3,  # 保留最近3轮对话
    return_messages=True
)

# 添加相同的对话
for role, message in conversations:
    if role == "用户":
        window_memory.chat_memory.add_user_message(message)
    else:
        window_memory.chat_memory.add_ai_message(message)

window_messages = window_memory.chat_memory.messages
print(f"滑动窗口存储的消息数: {len(window_messages)} (只保留最近3轮)")
for msg in window_messages:
    print(f"  {msg.type}: {msg.content[:50]}...")

# 4. 记忆加载到Prompt
print("\n4. 将记忆加载到Prompt")


def format_memory_for_prompt(memory):
    """格式化记忆为Prompt可用的格式"""
    messages = memory.chat_memory.messages

    formatted = []
    for msg in messages:
        if msg.type == "human":
            formatted.append(f"用户: {msg.content}")
        else:
            formatted.append(f"助手: {msg.content}")

    return "\n".join(formatted)


formatted_buffer = format_memory_for_prompt(buffer_memory)
formatted_window = format_memory_for_prompt(window_memory)

print("Buffer Memory格式化的Prompt:")
print(formatted_buffer[:200] + "...")
print(f"\n长度: {len(formatted_buffer)} 字符")

print("\nWindow Memory格式化的Prompt:")
print(formatted_window[:200] + "...")
print(f"长度: {len(formatted_window)} 字符")

# 5. 记忆管理策略对比
print("\n5. 记忆管理策略对比")


def simulate_conversation(memory, num_turns=20):
    """模拟多轮对话，记录记忆大小变化"""
    sizes = []

    for i in range(num_turns):
        # 添加用户消息
        memory.chat_memory.add_user_message(f"这是第{i + 1}轮对话的用户消息")
        # 添加助手回复
        memory.chat_memory.add_ai_message(f"这是第{i + 1}轮对话的助手回复")

        # 记录当前消息数量
        sizes.append(len(memory.chat_memory.messages))

    return sizes


# 测试不同记忆类型
buffer_sizes = simulate_conversation(ConversationBufferMemory(), 20)
window_sizes = simulate_conversation(ConversationBufferWindowMemory(k=6), 20)

# 可视化
plt.figure(figsize=(12, 6))
plt.plot(range(1, 21), buffer_sizes, 'o-', label='Buffer Memory (无限制)', linewidth=2)
plt.plot(range(1, 21), window_sizes, 's-', label='Window Memory (k=6)', linewidth=2)
plt.xlabel('对话轮数')
plt.ylabel('存储的消息数')
plt.title('不同记忆策略的内存使用对比')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('./memory_comparison.png', dpi=150)
plt.show()

print("Buffer Memory最终消息数: 40条 (20轮对话)")
print(f"Window Memory最终消息数: {window_sizes[-1]}条 (6轮对话 × 2)")

# 6. 清理记忆
print("\n6. 记忆清理")

buffer_memory.clear()
print(f"清理后Buffer Memory消息数: {len(buffer_memory.chat_memory.messages)}")

# 7. 记忆保存与加载
print("\n7. 记忆保存与加载")

# 保存记忆
saved_messages = buffer_memory.chat_memory.messages
print(f"当前记忆为空: {len(saved_messages)}")

# 添加一些对话
buffer_memory.chat_memory.add_user_message("你好，我想了解RAG")
buffer_memory.chat_memory.add_ai_message("RAG是检索增强生成技术")

# 导出记忆
exported_messages = []
for msg in buffer_memory.chat_memory.messages:
    exported_messages.append({
        'type': msg.type,
        'content': msg.content
    })

import json

with open('./saved_memory.json', 'w') as f:
    json.dump(exported_messages, f, indent=2)

print(f"已保存 {len(exported_messages)} 条消息到 learn12/saved_memory.json")

# 8. 记忆最佳实践
print("\n8. 记忆管理最佳实践")

best_practices = """
┌─────────────────────────────────────────────────────────────────┐
│ 场景                    │ 推荐记忆类型           │ 参数建议      │
├─────────────────────────────────────────────────────────────────┤
│ 客服对话（短）          │ BufferMemory          │ 无限制        │
│ 技术支持（长）          │ BufferWindowMemory    │ k=5-10        │
│ 复杂推理任务            │ SummaryMemory         │ max_token_limit=2000 │
│ 长文档问答              │ SummaryBufferMemory   │ k=4 + max_token=1000 │
└─────────────────────────────────────────────────────────────────┘

注意事项:
- 记忆太大会超过LLM的上下文限制
- 定期清理过期记忆
- 敏感信息需要脱敏处理
- 支持多会话隔离
"""
print(best_practices)

# 9. 保存记忆配置
memory_config = {
    'buffer_memory_count': len(buffer_memory.chat_memory.messages),
    'window_memory_k': 3,
    'window_memory_count': len(window_memory.chat_memory.messages),
    'best_practices': "使用WindowMemory处理长对话，使用BufferMemory处理短对话"
}

with open('./memory_config.json', 'w') as f:
    json.dump(memory_config, f, indent=2)

print("\n记忆配置已保存: learn12/memory_config.json")
print("\n✅ 对话记忆与多轮对话完成")