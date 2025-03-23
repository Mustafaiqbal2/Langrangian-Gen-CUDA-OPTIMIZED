class Embeddings:
    def __init__(self, vocab_size, embedding_dim):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.embeddings = self.create_embeddings()

    def create_embeddings(self):
        import torch
        return torch.nn.Embedding(self.vocab_size, self.embedding_dim)

    def forward(self, input_tokens):
        return self.embeddings(input_tokens)