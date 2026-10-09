
class tokenizer:
    def __init__(self, text):
        text = list(text)
        chars = sorted(list(set(text)))
        self.stoi = {ch : i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        self.vocab_size = len(chars)

    def encode(self, text):
        return [self.stoi[ch] for ch in text]
    def decode(self, indices):
        return ''.join([self.itos[i] for i in indices])
    
