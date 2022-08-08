import numpy as np
from utils import initialize_array
from loss import CrossEntropyLoss
from optimizers import SGD

class NormalizationLayer:
    """
    The normalization layer
    The code in this layer was directly sourced from the following link:
    https://github.com/renan-cunha/BatchNormalization/blob/master/src/feed_forward/layers.py
    """
    def __init__(self, n_dimension, optimizer, epsilon=1e-6, id=0):
        self.n_dimension = n_dimension
        self.gamma =initialize_array(1, n_dimension) # Element scale
        self.bias = initialize_array(1, n_dimension) # Element offset
        self.X = []
        self.mv = []
        self.running_mean_x = np.zeros(n_dimension)
        self.running_var_x = np.zeros(n_dimension)
        self.var_x = np.zeros(0)
        self.mean_x = np.zeros(0)
        self.epsilon = epsilon
        self.stddev_x = np.zeros(0)
        self.x_minus_mean = np.zeros(0)
        self.standard_x = np.zeros(0)
        self.optimizer = optimizer
        self.training = True
        self.running_avg_gamma = 0.9
        self.num_examples = 0
        self.gamma_grad = np.zeros(n_dimension)
        self.bias_grad = np.zeros(n_dimension)
        self.id = id
        self.optimizer.add(f'{self.id}-gamma', self.gamma)
        self.optimizer.add(f'{self.id}-bias', self.bias)

    def train(self):
        self.training = True

    def forward(self, x:np.ndarray):
        """
        Forward pass of the normalization layer.
        """
        self.bias = self.optimizer.get(f'{self.id}-bias')
        self.gamma = self.optimizer.get(f'{self.id}-gamma')
        
        
        self.mean_x = np.mean(x, axis=0)
        std_val = np.std(x, axis=0)
        
        self.var_x  = 1. / (std_val + self.epsilon)
        
        self.x_minus_mean = x - self.mean_x
        
        self.standard_x = self.x_minus_mean * self.var_x
        
        array = self.gamma * self.standard_x + self.bias
        
        return array
        

    def backward(self, grad_input: np.ndarray) -> np.ndarray:
        
        invN = 1. / np.prod(self.mean_x.shape)
        self.gamma_grad = np.sum(grad_input * self.standard_x, axis=(0,1))
        self.bias_grad = np.sum(grad_input, axis=(0, 1))

        standard_grad = grad_input * self.gamma
        
        var_grad = np.sum(standard_grad * self.x_minus_mean * -0.5 * self.var_x ** (-3/2),
                          axis=0, keepdims=True)
        
        
        mean_grad = np.mean(standard_grad * -self.var_x, axis=0,
                            keepdims=True)

      
        self.optimizer.update(f'{self.id}-gamma', self.gamma_grad)
        self.optimizer.update(f'{self.id}-bias',self.bias_grad)

        return standard_grad * self.var_x + var_grad * 2 * self.x_minus_mean * invN + mean_grad * invN
        
    def __repr__(self):
        return f'NormalizationLayer(n_dimension={self.n_dimension})'
        

if __name__ == '__main__':
    d_primary_size = 768
    batch = 7
    n_heads = 16
    seq_length = 17
    optimizer = SGD(0.001)
    norm_layer = NormalizationLayer(d_primary_size, optimizer)
    inputs = np.random.randn(batch, seq_length, d_primary_size)
    targets = np.random.randn(batch, seq_length, d_primary_size)
    inputs =  norm_layer.forward(inputs)
    loss = CrossEntropyLoss()
    gradient = loss.backward(inputs, targets)
    norm_layer.backward(gradient)
    