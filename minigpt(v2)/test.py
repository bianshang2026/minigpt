import torch
import torch.nn.functional as F

from model.minigpt import minigpt

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


model.train()

for step in range(10000):

   
    batch_indices = torch.randint(
        0,
        len(train_pairs),
        (320,)
    )

    x = train_pairs[batch_indices]

    targets = train_targets[batch_indices]



    optimizer.zero_grad()



    logits, _ = model(x)




    logits = logits[:, -1, :]

    loss = F.cross_entropy(
        logits,
        targets
    )
    loss.backward()
    optimizer.step()



    if step % 5000 == 0:

        prediction = logits.argmax(dim=-1)

        accuracy = (
            prediction == targets
        ).float().mean()

        print(
            
        )




model.eval()

with torch.no_grad():

    

    logits, _ = model(test_pairs)

    # [20, 2, 10]
    logits = logits[:, -1, :]

    # [20, 10] → [20]
    prediction = logits.argmax(dim=-1)




test_accuracy = (
    prediction == test_targets
).float().mean()


