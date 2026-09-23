import torch
import torch.nn as nn
from Transfomersblock import Transfomersblock



class minigpt(nn.Module):
    def __init__(self, d_model, vocab_size, block_size,numlayers = 4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_embedding = nn.Parameter(torch.zeros(1, block_size, d_model))
        self.blocks = nn.ModuleList([
            Transfomersblock(d_model)
            for _ in range(numlayers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        

    def forward(self, x, targets=None):
        seq_length = x.shape[1]
        position_embeddings = self.positional_embedding[:, :seq_length, :]
        embeddings = self.embedding(x) + position_embeddings
        x = embeddings
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = self.cross_entropy_loss(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss



model = minigpt(
    d_model=16,
    vocab_size=10000,
    block_size=10,
    numlayers=4
)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

model.train()

for step in range(5000):

    # 每次生成新的训练数据
    x = torch.randint(0, 10, (32, 10))

    # 真正的规律
    targets = (x + 1) % 10

    optimizer.zero_grad()

    logits, loss = model(x, targets)

    loss.backward()
    optimizer.step()

    if step % 500 == 0:
        prediction = logits.argmax(dim=-1)
        accuracy = (prediction == targets).float().mean()

model.eval()

test_x = torch.randint(0, 10, (32, 10))
test_targets = (test_x + 3) % 100

with torch.no_grad():
    test_logits, _ = model(test_x)
    prediction = test_logits.argmax(dim=-1)

test_accuracy = (prediction == test_targets).float().mean()

print("Test accuracy:", test_accuracy.item())


