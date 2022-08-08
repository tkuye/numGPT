"""Module for trainer and training the model."""
    
from optimizers import SGD
from utils import one_hot, batch_to_data
from loss import CrossEntropyLoss
import numpy as np
import os

class Trainer:
    """
    Base class for all trainers.
    """
    def __init__(self, model, dataset, batch_size=1, epochs=1, optimizer=None, save_path='./checkpoint'):
        self.model = model
        self.dataset = dataset
        self.batch_size = batch_size
        self.epochs = epochs
        self.loss = CrossEntropyLoss()
        if optimizer is None:
            self.optimizer = SGD()
        else:
            self.optimizer = optimizer
        
        self.num_batches = len(dataset) // batch_size
        if len(dataset) % batch_size != 0:
            self.num_batches += 1
        self.batch_index = 0
        self.save_path = save_path


    def train(self):
        print(f"Starting model training for {self.epochs} training iterations...")
         
        data_iter = iter(self.dataset)
        
        ten_percent = self.epochs // 10
        try:
            for i in range(self.epochs):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(self.dataset)
                    batch = next(data_iter)

                Y = self.dataset.shift(batch.copy())
                Y_data = batch_to_data(Y)
                
                X_hat = self.model.forward(batch.copy())
                batch_size = X_hat.shape[0]
                X_hat = X_hat.reshape(-1, X_hat.shape[-1])
                
                Y = one_hot(Y_data.reshape(-1), self.model.vocab_size)
                gradient = self.loss.backward(X_hat, Y)
            
                gradient = gradient.reshape(batch_size, self.model.max_len, self.model.vocab_size)
                self.model.backward(gradient)
                loss_val = np.sum(self.loss.forward(X_hat, Y)) / (batch_size * self.model.max_len)
                
                print(f"LOSS {i}:", loss_val)
                
                self.optimizer.step()

                if i % ten_percent == 0:
                    if not os.path.exists(self.save_path):
                        print('Model directory does not exist. Creating...')
                        try:
                            os.mkdir(self.save_path, 0o777)
                        except OSError:
                            print('Error creating model directory.')
                            exit(1)
                    path = os.path.join(self.save_path, f"{i}.pt")
                    self.model.save_model(path)
                    print("Saved model for epoch", i)
                
        except KeyboardInterrupt:
            print("Training interrupted.")
            path = os.path.join(self.save_path, f"{i}.pt")
            self.model.save_model(path)
            print("Saved model for epoch", i)
            return

        path = os.path.join(self.save_path, "model.pkl")
        self.model.save_model(path)
        print('Finished training.')
