"""Module for all the loss functions used by transformers."""
import numpy as np


class CrossEntropyLoss:
    def __init__(self, label_smoothing=0.0):
        self.items = None
        self.label_smoothing = label_smoothing

    def forward(self, X, y):
        """
        Forward pass of the cross entropy loss.
        """
        eps = np.finfo(float).eps

        # each example is associated with a single class; sum the negative log
        # probability of the correct label over all samples in the batch.
        # observe that we are taking advantage of the fact that y is one-hot
        # encoded

        cross_entropy = -np.sum(y * np.log(X + eps))
        return cross_entropy

    def item(self):
        return self.items

    def backward(self, X, y):
        
        # print('\n\nCED: ', np.where(y==1,-1/X, 0))
        grad = X - y
        return grad
    
    def __call__(self, y_hat, y):
        return self.forward(y_hat, y)



if __name__ == '__main__':
    loss = CrossEntropyLoss()
    y_hat = np.array([[0.1, 0.1, 0.8], [0.9, 0.05, 0.05]])
    y = np.array([[0, 0, 1], [1, 0, 0]])
    print(loss.backward(y_hat, y))
    

