class Encoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Encoder, self).__init__()
        self.input_dim = input_dim  
        self.output_dim = output_dim
       

    def forward(self, x):
        
        return x