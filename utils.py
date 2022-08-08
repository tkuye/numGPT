import numpy as np
from typing import List
from data import Input

def initialize_array(x, y):
    
    weights = np.random.randn(x,y) * np.sqrt(2.0 / x)
    return weights

def batch_dot(A, B):
    """
    Batch matrix multiplication
    """
    return np.einsum('ijk,ikj->ikj', A, B)


def bias_sum(grad, embedding_dim):
    return np.sum(grad, axis=(0, 1)).reshape(embedding_dim, 1)

def one_hot(arr:np.ndarray, vocab_size:int):
    """
    One-hot encoding of the input array.
    """
    return np.eye(vocab_size)[arr]

def batch_to_data(batch:List[Input]):
    """
    Convert a batch of data to a list of data.
    """
    return np.array([data.value for data in batch])