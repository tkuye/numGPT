"""Base Model for all transformer models."""


from optimizers import SGD
import pickle

class Base:
    def __init__(
    self, 
    vocab_size, 
    num_layers, 
    num_attn_heads, 
    embedding_dim,  
    hidden_dropout=0, 
    attn_dropout=0, 
    max_len=1024, 
    optimizer=None
    ):
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.num_attn_heads = num_attn_heads
        self.hidden_dropout = hidden_dropout
        self.attn_dropout = attn_dropout
        self.max_len = max_len
        self.embedding_dim = embedding_dim
        if optimizer is None:
            self.optimizer = SGD()
        else:
            self.optimizer = optimizer
        self.layers = []
        
    def forward(self, X):
        raise NotImplementedError
    
    def backward(self, d_out):
        raise NotImplementedError
    
    def __call__(self, x):
        return self.forward(x)

    def save_model(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load_model(filename) -> 'Base':
        with open(filename, 'rb') as f:
            return pickle.load(f)
        
        
    
    

    