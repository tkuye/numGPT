import numpy as np

class Dropout:
    def __init__(self, p=0):
        self.store_p = p
        self.p = 0
        self.mask = None
        self.train_mode = True
        
    def forward(self, x):
        if self.train_mode:
            self.mask = np.random.rand(*x.shape) < (1 - self.p)
            return (x * self.mask) / (1 - self.p)
        else:
            return x
        
    def backward(self, output_error):
        return output_error * self.mask

    def train(self):
        self.p = self.store_p
        self.train_mode = True
    def disable(self):
        self.train_mode = False
        
        

    def __call__(self, x, train_mode=True):
        return self.forward(x)
    
    def __repr__(self):
        return f'Dropout(p={self.p})'