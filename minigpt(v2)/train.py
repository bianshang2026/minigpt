import torch
import torch.nn as nn
from tokenizer import tokenizer
from model.minigpt import minigpt
with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()
tokenizer = tokenizer(text)
encoded = tokenizer.encode(text)
data = torch.tensor(encoded, dtype=torch.long)
def get_batch(data, batch_size, block_size):
    if len(data) <= block_size:
        raise ValueError(
            f"文本Token数量不足：len(data)={len(data)}, "
            f"block_size={block_size}，需要至少 {block_size + 1} 个Token"
        )

    ix = torch.randint(
        len(data) - block_size,
        (batch_size,)
    )

    x = torch.stack([
        data[i:i + block_size]
        for i in ix
    ])

    y = torch.stack([
        data[i + 1:i + block_size + 1]
        for i in ix
    ])

    return x, y

# 1. 设置训练参数
batch_size = 8
block_size = 128
d_model = 64
numlayers = 3
learning_rate = 0.01

# 2. 检查数据长度
print("Token 总数：", len(data))
print("词表大小：", tokenizer.vocab_size)

if len(data) <= block_size:
    raise ValueError("训练文本太短，请增加 input.txt 的内容")

# 3. 创建模型
model = minigpt(
    d_model=d_model,
    vocab_size=tokenizer.vocab_size,
    block_size=block_size,
    numlayers=numlayers
)

# 4. 创建优化器
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate
)

# 5. 开始训练
for step in range(10000):
    # 获取一批训练数据
    x, y = get_batch(data, batch_size, block_size)

   
    optimizer.zero_grad()

    
    logits, loss = model(x, y)

   
    loss.backward()

  
    optimizer.step()

   
    if step % 1000 == 0:
        print(f"Step {step}, Loss: {loss.item():.4f}")

print("训练完成！")
# 生成文本
model.eval()

start_text = text[:1]


start_ids = tokenizer.encode(start_text)
x = torch.tensor([start_ids], dtype=torch.long)


generated = model.generate(x, max_new_tokens=200)

result = tokenizer.decode(generated[0].tolist())

print("\n生成结果：")
print(result)