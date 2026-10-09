import torch
import torch.nn as nn
from model.Transfomersblock import Transfomersblock



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
        self.ffn = nn.Sequential(nn.Linear(d_model,d_model*4),nn.ReLU (),nn.Linear(d_model*4,d_model))
        self.apply(self._init_weights)        
    def _init_weights(self, module):
                    
                
                    if isinstance(module, nn.Linear):
                        nn.init.xavier_uniform_(module.weight)
                        if module.bias is not None:
                            nn.init.zeros_(module.bias)
                    elif isinstance(module, nn.Embedding):
                        nn.init.normal_(module.weight, mean=0, std=0.02)
                    elif isinstance(module, nn.LayerNorm):
                        nn.init.ones_(module.weight)
                        nn.init.zeros_(module.bias)
        

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

    @torch.no_grad()
    def generate(self, x, max_new_tokens, temperature=0.8, top_p=0.6):
     self.eval()

     for _ in range(max_new_tokens):
       
        x_cond = x[:, -self.positional_embedding.shape[1]:]

        
        logits, _ = self(x_cond)
        logits = logits[:, -1, :]

      
        temperature = max(temperature, 1e-5)
        logits = logits / temperature
        probs = torch.softmax(logits, dim=-1)

       
        sorted_probs, sorted_indices = torch.sort(
            probs, descending=True, dim=-1
        )

       
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        sorted_mask = cumulative_probs > top_p

       
        sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
        sorted_mask[..., 0] = False

        
        mask = torch.zeros_like(sorted_mask)
        mask.scatter_(-1, sorted_indices, sorted_mask)

       
        probs = probs.masked_fill(mask, 0.0)

       
        probs = probs / probs.sum(dim=-1, keepdim=True)

        
        next_token = torch.multinomial(probs, num_samples=1)

      
        x = torch.cat((x, next_token), dim=-1)

     return x

