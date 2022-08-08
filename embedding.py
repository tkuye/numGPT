import numpy as np
from utils import initialize_array
from data import Input
from typing import Union

class Embedding:
    """
    Embedding layer matrix used to turn vocab size into matrix table.
    """
    def __init__(self, input_dim:int, output_dim: int, init_scale: float = 0.02, optimizer=None):
        """
        Initialize the embedding layer.
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.init_scale = init_scale
        self.embedding_matrix = None
        self.inputs = []
        self.build()
        self.optimizer = optimizer
    
    def build(self):
        """
        Build the embedding matrix.
        """
        raise NotImplementedError()
    
    def forward(self, x):
        """
        Forward pass of the embedding layer.
        """
        raise NotImplementedError()

    def backward(self, d_out):
        """
        Backward pass of the embedding layer.
        """
        raise NotImplementedError()

    def get_weights(self):
        """
        Get the embedding matrix.
        """
        return self.embedding_matrix

    def get_position_encodings(self, sequence_length, d, n=10000):
        """
        Get the position encodings for a given position and depth.
        """
        P = np.zeros((sequence_length, d))
        for i in range(sequence_length):
            for j in np.arange(int(d/2)):
                denominator = np.power(n, 2 * j / d)
                P[i, 2 * j] = np.sin(i / denominator)
                P[i, 2 * j + 1] = np.cos(i / denominator)

        return P
    
    def set_weights(self, weights:np.ndarray):
        """
        Set the embedding matrix.
        """
        self.embedding_matrix = weights.copy()
        return self.embedding_matrix

class TokenEmbedding(Embedding):
    """
    Embedding layer for token embeddings.
    """

    def __init__(self, vocab:int, output_dim: int, init_scale: float = 0.02, optimizer=None):
        """
        Initialize the token embedding layer.
        """
        super(TokenEmbedding, self).__init__(vocab, output_dim, init_scale=init_scale, optimizer=optimizer)

    def build(self):
        """
        Build the embedding matrix.
        """
        
        self.embedding_matrix = initialize_array(self.output_dim, self.input_dim) * self.init_scale
        return self.embedding_matrix

    def forward(self, x):
        """
        Forward pass of the embedding layer.
        """
        self.inputs.append(x)
        
        return self.embedding_matrix[:, x]

    def backward(self, d_out):
        """
        Backward pass of the embedding layer.
        """
        delta = np.sum(d_out, axis=(0, 1))
        for token in self.inputs:
           
            self.embedding_matrix[:, token] = self.optimizer.update_step(self.embedding_matrix[:, token], delta, 'token')
        self.inputs = []
        return delta
    
    def __eq__(self, other):
        """
        Check if two embedding layers are equal.
        """

        return (self.embedding_matrix == other).all()


class PositionEmbedding(Embedding):
    """
    Embedding layer for position embeddings.
    """

    def __init__(self, sequence_length: int, embedding_dim: int, learnable=True, optimizer=None):
        """
        Initialize the position embedding layer.
        """
        self.learnable = learnable
        super(PositionEmbedding, self).__init__(embedding_dim, sequence_length, optimizer=optimizer)

    def build(self):
        """
        Build the embedding matrix.
        """
        if self.learnable:
            self.embedding_matrix = initialize_array(self.input_dim, self.output_dim) * self.init_scale
        else:
            self.embedding_matrix = self.get_position_encodings(self.input_dim, self.output_dim)
        return self.embedding_matrix

    def forward(self, x):
        """
        Forward pass of the embedding layer.
        """
        self.inputs.append(x)
        return self.embedding_matrix[:, x]

    def backward(self, d_out):
        """
        Backward pass of the embedding layer.
        """
        if self.learnable:
            delta = np.sum(d_out, axis=(0, 1))
            for token in self.inputs:
                self.embedding_matrix[:, token] = self.optimizer.update_step(self.embedding_matrix[:, token], delta, 'pos')
            self.inputs = []
            
        return delta


class GPTEmbedding:
    """An extra class to encapsulate behaviour of position and token embeddings. """

    def __init__(self, vocab_size:int, max_len:int,  embedding_dim: int, init_scale: float = 0.02, optimizer=None):
        """
        Initialize the GPTEmbedding.
        """

        self.token_embedding = TokenEmbedding(vocab_size, embedding_dim, init_scale=init_scale, optimizer=optimizer)
        self.position_embedding = PositionEmbedding(max_len, embedding_dim, optimizer=optimizer)
        self.optimizer = optimizer
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.embedding_dim = embedding_dim

    def forward(self, tokens):
        """
        Forward pass of the GPTEmbedding.
        """
        
        embeds = self.batch_input_embeds(tokens)
        
        return embeds.astype(float)

    def disable_positional_learning(self):
        """
        Disable positional learning.
        """
        self.position_embedding.learnable = False
        self.position_embedding.embedding_matrix = self.position_embedding.get_position_encodings(self.position_embedding.input_dim, self.position_embedding.output_dim)        

    def backward(self, d_out):
        """
        Backward pass of the GPTEmbedding.
        """
        self.token_embedding.backward(d_out)
        self.position_embedding.backward(d_out)

    def get_weights(self):
        """
        Get the embedding matrix.
        """
        return self.token_embedding.get_weights(), self.position_embedding.get_weights()

    def set_weights(self, weights:np.ndarray):
        """
        Set the embedding matrix.
        """
        self.token_embedding.set_weights(weights[0])
        self.position_embedding.set_weights(weights[1])

    def get_token_embedding(self, x):
        return self.token_embedding.forward(x)

    def get_position_embedding(self, x):
        return self.position_embedding.forward(x)

    def get_position_id(self, input_id:int, sequence:np.ndarray):
        """
        Get the position id for the input id.
        """
        return np.where(sequence == input_id)[0][0]
    
    # pylint: disable=unsubscriptable-object
    def build_input_embedding(self, position_id, sequence:Union[np.ndarray, Input]):
        """
        Build the input embedding.
        """
        if isinstance(sequence, np.ndarray):
            token_embedding = self.get_token_embedding(sequence[position_id])
        elif isinstance(sequence, Input):
            token_embedding = self.get_token_embedding(sequence.value[position_id])

        position_embedding = self.get_position_embedding(position_id)
        return token_embedding + position_embedding

    def build_input_embeds(self, sequence):
        """
        Build the input embeddings.
        """
        array = np.array([self.build_input_embedding(position_id, sequence) for position_id in range(len(sequence))])
        
        return array

    
    def batch_input_embeds(self, sequences):
        """
        Build the input embeddings.
        """
        array = np.array([self.build_input_embeds(sequence) for sequence in sequences], dtype=object)
        
        return array

    def __eq__(self, other):
        """
        Check if two embedding layers are equal.
        """
        return (self.token_embedding == other.token_embedding) and (self.position_embedding == other.position_embedding)

    def __call__(self, x):
        """
        Call the GPTEmbedding.
        """
        return self.forward(x)

    def __repr__(self):
        return f"GPTEmbedding(vocab_size={self.vocab_size}, max_len={self.max_len}, embedding_dim={self.embedding_dim})"

if __name__ == '__main__':
    vocab = 10000
    embedding = TokenEmbedding(vocab, output_dim=100)
    print(embedding.get_weights())