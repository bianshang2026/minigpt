import torch
import torch.nn as nn
from Selfattation import Selfattation

class Transfomersblock(nn.Module):

    def __init__(
        self,
    
        d_model,
    ):
        super().__init__()

        self.liner = nn.Linear(d_model, d_model)


        self.attention = Selfattation(input_dim=d_model, output_dim=d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model)
        )

       
    def forward(self, x):
     norm = self.norm1(x)
                
     attation_output = self.attention(norm)
                
     residual = x + attation_output
     layer2 = self.norm2(residual)
                           
                        
     ffn = self.ffn(layer2)


     output = residual + ffn
     return output