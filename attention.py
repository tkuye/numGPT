"""Module for the attention layer and mulithead attention layer."""
from torch.nn import MultiheadAttention as AttentionLayer
import numpy as np
import activations 
from dropout import Dropout
from utils import initialize_array, batch_dot, bias_sum
from loss import CrossEntropyLoss
from optimizers import Adam
from normalization import NormalizationLayer

class Attention:
    """
        Building a single masked self or cross attention layer.
    """
    def __init__(self, d_attention, d_primary_size, d_context_size,  out_size, optimizer):
        self.d_attention = d_attention
        self.d_primary_size = d_primary_size
        self.d_context_size = d_context_size
        self.out_size = out_size
        self.Wq = initialize_array(d_attention, d_primary_size)
        self.Wk = initialize_array(d_attention, d_context_size)
        self.Wv = initialize_array(out_size, d_context_size)
        self.Bq = initialize_array(d_attention, 1)
        self.Bk = initialize_array(d_attention, 1)
        self.Bv = initialize_array(out_size, 1)
        self.mask_S = None
        self.V = None
        self.K = None
        self.Q = None
        self.output = None
        self.primary_input = None
        self.context_input = None
        self.optimizer = optimizer

        self.S = None
        # Store the weights in a map for easy access.
        self.params = {
            'Wq': self.Wq,
            'Wk': self.Wk,
            'Wv': self.Wv,
            'Bq': self.Bq,
            'Bk': self.Bk,
            'Bv': self.Bv
        }

    def forward(self, primary_input, context_input, mask=None):
        # Q: (batch_size, d_attention, d_primary_size)
        # K: (batch_size, d_attention, d_context_size)
        # V: (batch_size, d_out, d_context_size)
        # mask: (batch_size, d_context_size)
        # output: (batch_size, d_out, d_attention)

        if mask is None:
            mask = np.ones((primary_input.shape[1], context_input.shape[1]))
        
        Q = np.dot(self.Wq, primary_input) + self.Bq
        K = np.dot(self.Wk, context_input) + self.Bk
        V = np.dot(self.Wv, context_input) + self.Bv
        S = np.dot(K.T, Q)

        self.V = V
        self.K = K
        self.Q = Q

        self.primary_input = primary_input
        self.context_input = context_input
        self.S = S
        mask_S = np.where(mask == 0, -np.inf, S)
        self.mask_S = mask_S
        
        output = activations.softmax(mask_S / np.sqrt(self.d_attention))
        self.output = output

        Vhat = np.dot(V, output)
        return Vhat

    
    def backward(self, d_out):

        deltaV = np.dot(self.output.T, d_out)
        delta = np.dot(d_out, self.V.T)
        delta = delta * activations.softmax_prime(self.output)
        
        # Not sure if this is needed. 
        deltaK = np.dot(delta, self.Q.T)
        deltaQ = np.dot(delta, self.K.T)

        dWq = np.dot(self.primary_input.T, deltaQ)
        dWk = np.dot(self.context_input.T, deltaK)
        dWv = np.dot(self.context_input.T, deltaV)

        dBq = np.sum(delta, axis=0)
        dBk = np.sum(delta, axis=0)
        dBv = np.sum(delta, axis=0)

        # Update the weights
        self.optimizer.add(self.Wq, dWq)
        self.optimizer.add(self.Wk, dWk)
        self.optimizer.add(self.Wv, dWv)
        self.optimizer.add(self.Bq, dBq)
        self.optimizer.add(self.Bk, dBk)
        self.optimizer.add(self.Bv, dBv)

        return delta

    
        

          
class MultiHeadAttention:
    """Computes Multi-Head (masked) self- or cross- attention layer.
    """
    def __init__(self, n_heads, embedding_dim,  dropout_rate=0, optimizer=None):
        self.n_heads = n_heads
        self.embedding_dim = embedding_dim
        
        self.Wo = initialize_array(embedding_dim, embedding_dim)
        self.Bo = initialize_array(embedding_dim, 1)
        self.dropout_rate = dropout_rate
        self.dropout = Dropout(dropout_rate)
        self.Wq = initialize_array(embedding_dim, embedding_dim)
        self.Wk = initialize_array(embedding_dim, embedding_dim)
        self.Wv = initialize_array(embedding_dim, embedding_dim)
        self.Bq = initialize_array(embedding_dim, 1)
        self.Bk = initialize_array(embedding_dim, 1)
        self.Bv = initialize_array(embedding_dim, 1)
        self.optimizer = optimizer
        self.optimizer.add('Wo', self.Wo)
        self.optimizer.add('Bo', self.Bo)
        self.optimizer.add('Wq', self.Wq)
        self.optimizer.add('Wk', self.Wk)
        self.optimizer.add('Wv', self.Wv)
        self.optimizer.add('Bq', self.Bq)
        self.optimizer.add('Bk', self.Bk)
        self.optimizer.add('Bv', self.Bv)

        self.curr_in = None
        self.S = None
        self.Q = None
        self.K = None
        self.V = None
        self.Y = None
        self.optimizer = optimizer
    

    def linear(self, W, x,  b=None):
        
        if b is not None:
            b = np.expand_dims(b, axis=1).T
            
            return np.dot(x, W.T) + b
        else:
            return np.dot(x, W.T)

    def train(self):
        self.dropout.train()
        
    def forward(self, in_features,  mask=None):
        self.Wq = self.optimizer.get('Wq')
        self.Wk = self.optimizer.get('Wk')
        self.Wv = self.optimizer.get('Wv')
        self.Wo = self.optimizer.get('Wo')
        self.Bo = self.optimizer.get('Bo')
        self.Bq = self.optimizer.get('Bq')
        self.Bk = self.optimizer.get('Bk')
        self.Bv = self.optimizer.get('Bv')

        q = self.linear(self.Wq, in_features, self.Bq)
        k = self.linear(self.Wk, in_features, self.Bk)
        v = self.linear(self.Wv, in_features, self.Bv)

        batch, seq, feature_size = q.shape
        d_attention = self.embedding_dim // self.n_heads

        q = q.reshape(batch, seq,  self.n_heads, d_attention).transpose(0, 2, 1, 3)\
            .reshape(batch*self.n_heads, seq, d_attention)
        k = k.reshape(batch, seq,  self.n_heads, d_attention).transpose(0, 2, 1, 3)\
            .reshape(batch*self.n_heads, seq, d_attention)
        v = v.reshape(batch, seq,  self.n_heads, d_attention).transpose(0, 2, 1, 3)\
            .reshape(batch*self.n_heads, seq, d_attention)
       
        att = np.einsum('ijk,kli->ijl', q, k.T)
        att = att / np.sqrt(self.embedding_dim)
        
        if mask is None:
            mask = np.ones((batch*self.n_heads, seq, seq))

        # Can't be set to -inf because otherwise we get subtraction overflow 
        mask_S = np.where(mask == 0, -1000000000, att)
        
        S = activations.softmax(mask_S, axis=1)
        
        
        Y = np.matmul(S, v)
        
        
        batch, seq, feature_size = Y.shape
        out_dim = feature_size * self.n_heads
        batch //= self.n_heads
        Y = Y.reshape(batch, self.n_heads, seq, feature_size)
        Y = Y.transpose(0, 2, 1, 3).reshape(batch, seq, out_dim)
        
        self.Q = q
        self.K = k
        self.V = v
        self.S = S
        self.curr_in = in_features
        
        Y = self.linear(self.Wo, Y, self.Bo)
        Y = self.dropout.forward(Y)
        self.Y = Y
        return Y

    def backward(self, grad):
        batch, seq, _ = grad.shape
        delta = batch_dot(self.Y.transpose(0, 2, 1), grad)
        
        dW = delta
        
        dB = bias_sum(grad, self.embedding_dim)
        dW = bias_sum(dW, self.embedding_dim)
        
        

        self.optimizer.update('Wo', dW)
        self.optimizer.update('Bo', dB)
            

    
        
        
        d_attention = self.embedding_dim // self.n_heads
        
        delta = delta.reshape(batch, seq,  self.n_heads, d_attention).transpose(0, 2, 1, 3)\
            .reshape(batch*self.n_heads, seq, d_attention)


        
        deltaV = np.einsum('ijk,ijj->ijk', delta, self.S.transpose(0, 2, 1))
        deltaV = delta.reshape(batch, seq, self.embedding_dim)
        deltaV = batch_dot(self.curr_in.transpose(0, 2, 1), deltaV)
        deltaBv = bias_sum(deltaV, self.embedding_dim)
        deltaV = bias_sum(deltaV, self.embedding_dim)
        

        self.optimizer.update('Wv', deltaV)


        self.optimizer.update('Bv', deltaBv)
        
    
        deltaK = np.einsum('ijk,ijj->ijk', delta, self.S)
        deltaK = np.einsum('ijk,ikj->ijk', deltaK, self.V.transpose(0, 2, 1))
        
        deltaK = np.einsum('ijk,ikj->ijk', deltaK, self.Q.transpose(0, 2, 1)/np.sqrt(self.embedding_dim))
        deltaK = np.reshape(deltaK, (batch, seq, self.embedding_dim))
        deltaBk = np.sum(deltaK, axis=(0, 1)).reshape(self.embedding_dim, 1)
        deltaK = np.sum(deltaK, axis=(0, 1)).reshape(self.embedding_dim, 1)
        
        self.optimizer.update('Wk', deltaK)
        
        self.optimizer.update('Bk', deltaBk)
        
        
        deltaQ = np.einsum('ijk,ijj->ijk', delta, self.S)
        
        deltaQ = np.einsum('ijk,ikj->ijk', deltaQ, self.V.transpose(0, 2, 1))
        
        deltaQ = np.einsum('ijk,ijk->ijk', deltaQ, self.K / np.sqrt(self.embedding_dim))
        
        deltaQ = np.reshape(deltaQ, (batch, seq, self.embedding_dim))
        
        deltaBq = np.sum(deltaQ, axis=(0, 1)).reshape(self.embedding_dim, 1)
        deltaQ = np.sum(deltaQ, axis=(0, 1)).reshape(self.embedding_dim, 1)

        self.optimizer.update('Wq', deltaQ)

        self.optimizer.update('Bq', deltaBq)

        delta = delta.reshape(batch, seq,  self.n_heads, d_attention).transpose(0, 2, 1, 3)\
            .reshape(batch, seq, d_attention*self.n_heads)
        return delta

    def __repr__(self) -> str:
        return f'MultiHeadAttention(n_heads={self.n_heads}, embedding_dim={self.embedding_dim})'


if __name__ == '__main__':
    # Test the attention layer.
    d_primary_size = 4
    batch = 1
    n_heads = 2
    seq_length = 2
    optimizer = Adam(1e-2)
    attention_layer = MultiHeadAttention(n_heads=n_heads, embedding_dim=d_primary_size, dropout_rate=1, optimizer=optimizer)
    attention_layer.train()
    inputs = np.random.randn(batch, seq_length, d_primary_size)
    targets = np.zeros_like(inputs)
    torch_att = AttentionLayer(d_primary_size, n_heads)
    for i in range(batch):
        for j in range(seq_length):
            idx = np.random.randint(0, seq_length)
            targets[i, j, idx] = 1
    
    norm_layer = NormalizationLayer(d_primary_size, optimizer=optimizer)
    inputs_in = attention_layer.forward(inputs)
    loss = CrossEntropyLoss()
    gradient = loss.backward(inputs, targets)
    i = 0
    while True:
        inputs_out = attention_layer.forward(inputs_in)
        
        #inputs_out = norm_layer.forward(inputs_out)
        gradient = loss.backward(inputs_out, targets)
        loss_val = np.sum(loss.forward(inputs_out, targets))
        #print("BEFORE GRAD_VIEW", gradient[0, 3, :])
        
        grad = attention_layer.backward(gradient)
        
        #grad = norm_layer.backward(grad)
        optimizer.step()
        
        print(i, loss_val)
        
        if np.isnan(loss_val) or np.isinf(loss_val):
            break
        i += 1
        
        
        

        
        
    
    
    
    
    