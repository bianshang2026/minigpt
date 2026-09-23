import torch
import torch.nn.functional as F

from minigpt import minigpt

torch.manual_seed(42)


# ============================================================
# 1. 创建模型
# ============================================================

model = minigpt(
    d_model=32,
    vocab_size=10000,
    block_size=2,
    numlayers=8
)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.01
)


# ============================================================
# 2. 创建全部 100 种数字组合
# ============================================================

pairs = []

for a in range(100):
    for b in range(100):
        pairs.append([a, b])

pairs = torch.tensor(pairs, dtype=torch.long)

print("所有组合数量:", len(pairs))
print("pairs shape:", pairs.shape)


# ============================================================
# 3. 划分训练集和测试集
# ============================================================

# 打乱 100 个组合
indices = torch.randperm(len(pairs))

train_size = 8000

train_indices = indices[:train_size]
test_indices = indices[train_size:]

train_pairs = pairs[train_indices]
test_pairs = pairs[test_indices]


# ============================================================
# 4. 根据规则生成答案
#
# y = (a + b) % 10
# ============================================================

train_targets = (
    train_pairs[:, 0] +
    train_pairs[:, 1]
) % 10

test_targets = (
    test_pairs[:, 0] +
    test_pairs[:, 1]
) % 10


print()
print("训练集:")
print(train_pairs)

print()
print("训练答案:")
print(train_targets)

print()
print("测试集:")
print(test_pairs)

print()
print("测试答案:")
print(test_targets)


# ============================================================
# 5. 开始训练
# ============================================================

model.train()

for step in range(10000):

    # 从 80 个训练组合中随机抽 32 个
    batch_indices = torch.randint(
        0,
        len(train_pairs),
        (320,)
    )

    x = train_pairs[batch_indices]

    targets = train_targets[batch_indices]


    # --------------------------------------------------------
    # 清空上一轮梯度
    # --------------------------------------------------------

    optimizer.zero_grad()


    # --------------------------------------------------------
    # 前向传播
    #
    # x:
    # [B, 2]
    #
    # logits:
    # [B, 2, 10]
    # --------------------------------------------------------

    logits, _ = model(x)


    # --------------------------------------------------------
    # 我们只看最后一个位置
    #
    # logits:
    # [B, 2, 10]
    #
    # logits[:, -1, :]:
    # [B, 10]
    # --------------------------------------------------------

    logits = logits[:, -1, :]


    # --------------------------------------------------------
    # 计算 loss
    # --------------------------------------------------------

    loss = F.cross_entropy(
        logits,
        targets
    )


    # --------------------------------------------------------
    # 反向传播
    # --------------------------------------------------------

    loss.backward()


    # --------------------------------------------------------
    # 更新参数
    # --------------------------------------------------------

    optimizer.step()


    # --------------------------------------------------------
    # 打印训练情况
    # --------------------------------------------------------

    if step % 5000 == 0:

        prediction = logits.argmax(dim=-1)

        accuracy = (
            prediction == targets
        ).float().mean()

        print(
            f"Step: {step:4d} | "
            f"Loss: {loss.item():.6f} | "
            f"Train Accuracy: {accuracy.item():.4f}"
        )


# ============================================================
# 6. 测试
# ============================================================

model.eval()

with torch.no_grad():

    # --------------------------------------------------------
    # 一次性把所有没见过的组合送进去
    # --------------------------------------------------------

    logits, _ = model(test_pairs)

    # [20, 2, 10]
    logits = logits[:, -1, :]

    # [20, 10] → [20]
    prediction = logits.argmax(dim=-1)


# ============================================================
# 7. 计算测试准确率
# ============================================================

test_accuracy = (
    prediction == test_targets
).float().mean()


print()
print("=" * 60)
print("测试结果")
print("=" * 60)

print(
    f"Test Accuracy: {test_accuracy.item():.4f}"
)


# ============================================================
# 8. 把每一个测试组合打印出来
# ============================================================

print()
print("具体预测:")
print("-" * 60)

for i in range(len(test_pairs)):

    a = test_pairs[i, 0].item()
    b = test_pairs[i, 1].item()

    pred = prediction[i].item()
    target = test_targets[i].item()

    correct = "✓" if pred == target else "✗"



# ============================================================
# 9. 最终解释
# ============================================================

print()
print("=" * 60)

if test_accuracy.item() == 1.0:

    print("测试集全部正确！")
    print()
    print("模型成功泛化到了训练过程中没有出现过的组合。")

elif test_accuracy.item() >= 0.8:

    print("模型具有比较明显的组合泛化能力。")

elif test_accuracy.item() >= 0.5:

    print("模型有一定泛化能力，但还没有完全学会这个关系。")

else:

    print("模型没有很好地泛化到没见过的组合。")

print("=" * 60)