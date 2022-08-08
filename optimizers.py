"""Module for optimizers."""
from typing import Dict, List
import numpy as np


class Optimizer:
    """
    Base class for all optimizers.
    """
    def __init__(self, lr=0.01, clip_norm_range = 2):
        self.lr = lr
        self.pg:Dict[List] = {}
        self.num_params = 0
        self.clip_norm_val = clip_norm_range

    def step(self):
        """
        Update the weights.
        """
        raise NotImplementedError
    
    def add(self, key, param, grad=None):
        """
        Add a parameter and its gradient to the optimizer.
        """
        self.pg[key] = [param, grad]
        self.num_params += 1

    def keys(self):
        """
        Get the keys of the optimizer.
        """
        return self.pg.keys()

    def update(self, key, grad):
        """
        Update the gradient of a parameter.
        """
        self.pg[key][1] = grad

    def update_embed(self, key, embed, grad):
        """
        Update the embedding of a parameter.
        """
        self.pg[key][0] = embed
        self.pg[key][1] = grad
    
    def clip_norm(self, grad):
        """
        Clip the norm of the gradient.
        """
        if self.clip_norm_val is not None:
            grad = np.clip(grad, -1*self.clip_norm_val, self.clip_norm_val)
        return grad

    def get_params(self):
        """
        Get the parameters of the optimizer.
        """
        return self.pg.values()

    def get_weights(self):
        """
        Get the weights of the optimizer.
        """
        return self.pg.items()

    def load_weights(self, weights):
        """
        Load the parameters of the optimizer.
        """
        for key, weight in weights.items():
            self.pg[key] = weight

    def __getitem__(self, key):
        """
        Get the parameter of the optimizer.
        """
        return self.pg.get(key)[0]
    def get(self, key):
        """
        Get the parameter of the optimizer.
        """
        return self.pg.get(key)[0]

    def drop(self, key):
        """
        Drop a parameter from the optimizer.
        """
        del self.pg[key]
        self.num_params -= 1

    def update_step(self, param, grad, key):
        """
        Update the step of the optimizer.
        """
        raise NotImplementedError
   

class SGD(Optimizer):
    def __init__(self, lr=0.01 , clip_norm_range = 2):
        super(SGD, self).__init__(lr, clip_norm_range)
        self.lr = lr
        
    def step(self):
        for key, (param, grad) in self.pg.items():
            param -= self.lr * grad
            self.pg[key] = [param, grad]
           
    def update_step(self, param, grad):
        grad = self.clip_norm(grad)
        param -= self.lr * grad
        return param
        
        
    
class Adam(Optimizer):
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, clip_norm_range = 2):
        super(Adam, self).__init__(lr, clip_norm_range)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = None
        self.v = None
        self.t = 0
        self.m_embed, self.v_embed = {}, {}
        
    def step(self):

        if self.m is None:
            self.m, self.v = [], []
            for param in self.pg.values():
                self.m.append(np.zeros_like(param[0]))
                self.v.append(np.zeros_like(param[0]))
        self.t += 1
        lr_t = self.lr * np.sqrt(1.0 - self.beta2 ** self.t) / (1.0 - self.beta1 ** self.t)
        for i, (key, value) in enumerate(self.pg.items()):
            param, grad = value
            grad = self.clip_norm(grad)
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * grad ** 2
            param -= lr_t * self.m[i] / (np.sqrt(self.v[i]) + self.eps)
            
            self.pg[key] = [param, grad]
        
    def update_step(self, param, grad, key):
        if self.m_embed.get(key) is None:
            m_param = np.zeros_like(param)
            v_param = np.zeros_like(param)
        else:
            m_param = self.m_embed[key]
            v_param = self.v_embed[key]
        self.t += 1
        lr_t = self.lr * np.sqrt(1.0 - self.beta2 ** self.t) / (1.0 - self.beta1 ** self.t)
        grad = self.clip_norm(grad)
        m_param = self.beta1 * m_param + (1 - self.beta1) * grad
        v_param = self.beta2 * v_param + (1 - self.beta2) * grad ** 2
        param -= lr_t * m_param / (np.sqrt(v_param) + self.eps)

        self.m_embed[key] = m_param
        self.v_embed[key] = v_param
        
        return param



if __name__ == '__main__':
    np.random.randn(3, 4)
    optimizer = Adam(lr=0.01)
    np.random.seed(0)
    params = np.random.randn(3, 4)
    grads = np.random.randn(3, 4)
    optimizer.add('key', params, grads)
    optimizer.step()
    