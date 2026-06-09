import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,  #Causal LM = 自回归语言模型
    AutoModelForMaskedLM,  #掩码语言模型
    pipeline,
    set_seed
)
import matplotlib.pyplot as plt
import numpy as np

print("=" * 60)
print("文本生成和掩码填充")
print("=" * 60)

# 设置随机种子
set_seed(42)

# 1. 文本生成（GPT-2）
print("\n1. 文本生成 (GPT-2)")

model_name = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# 添加Padding token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"模型：{model_name}")
print(f"模型参数:{sum(p.numel() for p in model.parameters()):,}")

def generate_text(prompt,max_length=100,temperature=0.7,top_k=50,top_p=0.9):
    """文本生成"""
    inputs = tokenizer(prompt,return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(              #Hugging Face 提供的文本生成接口
            inputs.input_ids,
            max_length=max_length,
            temperature=temperature,               #随机性控制
            top_k=top_k,                           #限制每一步生成时只从概率最高的前 K 个 token 选择，避免低概率 token
            top_p=top_p,                           #核采样 / nucleus sampling）：选择累计概率达到 p 的 token 集合
            do_sample=True,                        #使用采样而非贪心策略
            pad_token_id = tokenizer.eos_token_id  #填充 token ID，避免生成时报错
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

# 不同温度下的生成
prompt = "The future of artificial intelligence is"
print(f"提示: {prompt}")
print("\n不同温度下的生成结果:")
for temp in [0.2,0.7,1.2]:
    generated = generate_text(prompt,temperature=temp,max_length=80)
    print(f"\n温度 {temp}:")
    print(f"  {generated}")

# 2. 使用Pipeline进行文本生成
print("\n2. Pipeline文本生成")

generator = pipeline("text-generation",model=model_name,tokenizer=tokenizer)

prompts = [
    "Once upon a time, there was a",
    "The best way to learn programming is",
    "In the future, robots will"
]

for prompt in prompts:
    results = generator(prompt, max_length=60, num_return_sequences=1)
    print(f"\n提示: {prompt}")
    print(f"生成: {results[0]['generated_text']}")

# 掩码填充
print("\n3.掩码填充（Masked Language Model）")

mlm_model_name = 'bert-base-uncased'
mlm_tokenizer = AutoTokenizer.from_pretrained(mlm_model_name)
mlm_model = AutoModelForMaskedLM.from_pretrained(mlm_model_name)

def fill_mask(text,top_k=5):
    """填充[MASK]位置"""
    inputs = mlm_tokenizer(text,return_tensors='pt')  #将文本 text 转换成 PyTorch 张量  返回input_ids，attention_mask

    with torch.no_grad():
        outputs = mlm_model(**inputs)
        logits = outputs.logits

    mask_index = inputs['input_ids'][0].tolist().index(mlm_tokenizer.mask_token_id)
    probs = torch.softmax(logits[0,mask_index],dim=-1)

    top_probs,top_indices = torch.topk(probs,top_k)

    print(f"输入：{text}")
    print(f"预测结果")
    for prob,idx in zip(top_probs,top_indices):
        token = mlm_tokenizer.decode([idx])
        print(f" {token}:{prob.item():.3f}")

# 测试MLM
fill_mask("The capital of France is [MASK].")
fill_mask("I love to read [MASK] books.")
fill_mask("The [MASK] is shining brightly in the sky.")
fill_mask("She is a [MASK] doctor who saved many lives.")

