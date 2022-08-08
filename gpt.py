"""
Where the GPT based model will be implemented.
"""

import numpy as np
from tokenizer import tokenizer
from activations import gelu, softmax, gelu_prime
from normalization import NormalizationLayer
from attention import MultiHeadAttention
from dropout import Dropout
from base import Base
from utils import initialize_array
from optimizers import Adam
from loss import CrossEntropyLoss
from embedding import GPTEmbedding
from data import create_input_from_string
import utils


class GPT(Base):
    """
    GPT based architecture nin which a language model can be sufficiently trained. 
    Refer to the following papers for more information:
    1. https://arxiv.org/pdf/2207.09238.pdf
    2. https://arxiv.org/pdf/1706.03762.pdf
    3. https://arxiv.org/pdf/2005.14165.pdf
    """
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
        super(GPT, self).__init__(
        vocab_size, num_layers,
        num_attn_heads, embedding_dim,  
        hidden_dropout, attn_dropout, 
        max_len, optimizer)
        self.optimizer = optimizer
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.layers.append(GPTEmbedding(vocab_size, max_len, embedding_dim, optimizer=optimizer))
        for i in range(num_layers):
            self.layers.append(GPTLayer(embedding_dim, embedding_dim, self.optimizer, num_attn_heads, attn_dropout=attn_dropout, layer_number=i))
            self.layers.append(Dropout(hidden_dropout))
        
        self.Wu = initialize_array(vocab_size, embedding_dim)
        self.layers.append(NormalizationLayer(embedding_dim, optimizer, id=np.random.randint(0, 100)))
        self.X = None
        self.optimizer.add('Wu', self.Wu)
    

    def __call__(self, x):
        return self.forward(x)
    
    def forward(self, X):
        
        for layer in self.layers:
            X = layer.forward(X)
        
        self.X = X
        
        Z = np.dot(X, self.Wu.T)
        
        P = softmax(Z, axis=2)
        
        return P

    def disable_positional_learning(self):
    
        self.layers[0].disable_positional_learning()

    def backward(self, d_out):
        dWu = np.einsum('abc,acd->acd',self.X.transpose(0, 2, 1), d_out)
        dWu = np.sum(dWu, axis=(0, 1)).reshape(self.vocab_size,  1)
        
        self.optimizer.update('Wu', dWu)
    
        delta = np.einsum('abc,dc->abd', d_out, self.Wu.T)
    
        for layer in reversed(self.layers):
            delta = layer.backward(delta)
        return delta

    def __repr__(self):
        rep = 'GPT Model(\n'
        for layer in self.layers:
            rep += str(layer) + '\n'
        rep += ')'
        return rep

    def generate(self, input_ids, seq_len, temperature=1.0, top_k:int=0):
        """
        Generate text from the model.
        """
        # Turn off our dropout layers
        for layer in self.layers:
            if isinstance(layer, Dropout):
                layer.disable()
            if isinstance(layer, GPTLayer):
                layer.MultiHeadAttention.dropout.disable()
        
        
        for _ in range(seq_len):
            logits = self.forward(np.array([input_ids]))
            if temperature != 0:
                logits = logits / temperature
            probs = softmax(logits, axis=2)
            if top_k == 0:
                next_id = np.argmax(probs, axis=-1)[0][0]
            else:
                inputs = np.argsort(probs, axis=2)[:, :, -top_k:][0][0]

                probs = softmax(probs[:, :, -top_k:][0][0])
                next_id = np.random.choice(inputs, 1, p=probs)
            
            input_ids = np.append(input_ids, next_id)
        # Return text
        return input_ids

    

class GPTLayer:
    """
    Create a GPT based layer. 
    """
    def __init__(self, hidden_size, n_dimension, optimizer, n_heads=12, attn_dropout=0, layer_number=0) -> None:
        self.hidden_size = hidden_size
        self.n_dimension = n_dimension 
        self.n_heads = n_heads
        self.output_dim = n_dimension
        # Our dropout components
        self.attn_dropout = attn_dropout
        
        self.W1 = initialize_array(self.hidden_size, self.n_dimension)
        self.W2 = initialize_array(self.n_dimension, self.hidden_size)
        self.B1 = np.zeros((self.hidden_size, 1))
        self.B2 = np.zeros((self.n_dimension, 1))
        
        self.Xhat = None
        self.X = None
        self.optimizer = optimizer
        self.optimizer.add('W1', self.W1)
        self.optimizer.add('W2', self.W2)
        self.optimizer.add('B1', self.B1)
        self.optimizer.add('B2', self.B2)
        # Our normalization components
        self.layer_norm1 = NormalizationLayer(self.n_dimension, optimizer, id=np.random.randint(0, 100) + layer_number)
        self.layer_norm2 = NormalizationLayer(self.n_dimension, optimizer, id=np.random.randint(0, 100) + layer_number)
        
        # For GPT, our embedding dim is the same for attention, primary and output.
        self.MultiHeadAttention = MultiHeadAttention(n_heads=self.n_heads, embedding_dim=self.n_dimension,dropout_rate=self.attn_dropout, optimizer=self.optimizer)

    def forward(self, X):
        """
        Forward pass of the layer.
        """
        self.W1 = self.optimizer.get('W1')
        self.W2 = self.optimizer.get('W2')
        self.B2 = self.optimizer.get('B2')
        self.B1 = self.optimizer.get('B1')
        Xhat = X.copy()
        
        mask = np.ones((X.shape[1], X.shape[1]))
        mask = np.tril(mask)
        
        Xhat = self.layer_norm1.forward(X)
       # Note that mask needs to come from outer function as we need to pass the status due to the next token.
        X = X + self.MultiHeadAttention.forward(Xhat, mask=mask)
        Xhat = self.layer_norm2.forward(X)
        self.Xhat = Xhat
        X = X + self.linear(gelu(self.linear(Xhat, self.W1, self.B1)), self.W2, self.B2)
        
        return X

    def linear(self, X, W, B=None):
        if B is not None:
            B = np.expand_dims(B, axis=1).T
            return np.dot(X, W.T) + B
        return np.dot(X, W.T)
    
    def backward(self, d_out):
        """
        Backward pass of the layer.
        """
        # Debatable dot with itself.
        gelu_grad = self.linear(self.Xhat, self.W1, self.B1)
        gelu_grad  = gelu(gelu_grad)
        
        
        dW2 = np.einsum('abc,acb->acb', gelu_grad.transpose(0, 2, 1), d_out)
        
        delta = np.dot(d_out, self.W2.T)
        
        delta = delta * gelu_prime(gelu_grad)
        
        dW1 = np.einsum('abc,acb->acb', self.Xhat.transpose(0, 2, 1), delta)
        dB2 = np.sum(d_out, axis=1, keepdims=True)
        dB1 = np.sum(delta, axis=1, keepdims=True)
        
        # Update the weights
    
        delta = self.layer_norm2.backward(delta)
        delta = self.MultiHeadAttention.backward(delta)
        delta = self.layer_norm1.backward(delta)

        dW1 = np.sum(dW1, axis=(0, 1)).reshape(self.hidden_size, 1)
        dW2 = np.sum(dW2, axis=(0, 1)).reshape(self.hidden_size, 1)
        dB1 = np.sum(dB1, axis=(0, 1)).reshape(self.hidden_size, 1)
        dB2 = np.sum(dB2, axis=(0, 1)).reshape(self.hidden_size, 1)
        self.optimizer.update('W1', dW1)
        self.optimizer.update('W2', dW2)
        self.optimizer.update('B1', dB1)
        self.optimizer.update('B2', dB2)

        return delta

    def __repr__(self):
        return f"""GPTLayer(
            {self.layer_norm1}
            {self.MultiHeadAttention}
            {self.layer_norm2}
            Linear({self.hidden_size}, {self.n_dimension})
        )"""


if __name__ == '__main__':
    
    tok = tokenizer()
    vocab_size = len(tok.vocab)
    
    gpt = GPT(vocab_size=tok.vocab.size, embedding_dim=16, num_layers=4, num_attn_heads=4, max_len=6, hidden_dropout=0.8, attn_dropout=0.9, optimizer=Adam(lr=5e-3))
   
    input_seq = create_input_from_string('hello, what is your', tokenizer=tok, max_len=6, shiftable=False)
    
    input_arr = np.array([input_seq])
    
    d_primary_size = 16
    batch = 1
    seq_length = len(input_seq)
    
    y = create_input_from_string(', what is your name', tokenizer=tok, max_len=6, shiftable=False)

    y_arr = np.array([y])

    y_arr = utils.one_hot(y_arr.reshape(-1), vocab_size)
    
    loss = CrossEntropyLoss()
    
    i = 0
    while i < 500:
        y_hat = gpt.forward(input_arr)
       
        y_hat = y_hat.reshape(-1, y_hat.shape[-1])
        
        gradient = loss.backward(y_hat, y_arr)
        
        gradient = gradient.reshape(batch, seq_length, vocab_size)
        
        gpt.backward(gradient)
        loss_val = np.sum(loss.forward(y_hat, y_arr))
        print(f"LOSS {i}:", loss_val)
        
        gpt.optimizer.step()
        i += 1
    

    seq = create_input_from_string('Hello, what is your', tokenizer=tok, max_len=6)
    test_input = np.array([seq])
    token = gpt.forward(test_input)
    token = token.reshape(-1, token.shape[-1])
    token_m = np.argmax(token, axis=-1)
    token_m =  token_m[0]
    toks = tok.decode([token_m])
    print(f"Hello, what is your{toks}")
    print("Probs:", np.max(token, axis=-1)[0])
    gpt.save_model('gpt.pkl')
    
        
        



    
    

    
    
 