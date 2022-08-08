"""Module for all data related and processing activities."""
import numpy as np 
from typing import List
from tokenizer import tokenizer
from copy import deepcopy

class Input(object):
    def __init__(self, input_ids, shiftable=True):
        if shiftable:
            self.input_ids = np.array(input_ids[:-1])
            self.shifted = False
            self.final_idx = input_ids[-1]
            self.__input_len = len(input_ids[:-1])
        else:
            self.input_ids = np.array(input_ids)
            self.shifted = True
            self.final_idx = None
            self.__input_len = len(input_ids)
        
    def __len__(self):
        return self.__input_len

    def __getitem__(self, index):
        return self.input_ids[index]   

    @property
    def value(self):
        return self.input_ids

    @value.setter
    def value(self, value):
        self.input_ids = value
        self.__input_len = len(value)

    def shift(self):
        """
        Shift the input by one position.
        """
        copy_self = deepcopy(self)
        if not copy_self.shifted:
            
            copy_self.input_ids = np.append(copy_self.input_ids[1:],  copy_self.final_idx)
            copy_self.shifted = True
            
        return copy_self

    
class Dataset(object):
    """
    Base class for all datasets.
    """

    def __init__(self, data:List[Input], batch_size=1):
        self.data = data
        self.batch_size = batch_size
        self.num_batches = len(data) // batch_size
        self.batch_index = 0
        if len(data) % batch_size != 0:
            self.num_batches += 1
        self.shuffle()
        self.shifted = False

    def shuffle(self):
        """
        Shuffle the data.
        """
        np.random.shuffle(self.data)

    def add(self, data):
        """
        Add data to the dataset.
        """
        self.data.append(data)
        self.num_batches = len(self.data) // self.batch_size
        if len(self.data) % self.batch_size != 0:
            self.num_batches += 1
    
    def shift(self, batch):
        """
        Shift the batch.
        """
        new_batch = []
        for data in batch:
            d = data.shift()
            new_batch.append(d)
        return new_batch

    def __getitem__(self, index):
        """
        Get the data at the given index.
        """
        return self.data[index]

    def __len__(self):
        """
        Get the length of the dataset.
        """
        return len(self.data)
    
    def __iter__(self):
        """
        Iterate over the dataset.
        """
        return self

    def __next__(self):
        """
        Get the next batch of data.
        """
        if self.batch_index >= self.num_batches:
            self.batch_index = 0
            self.shuffle()
        batch = self.data[self.batch_index * self.batch_size: (self.batch_index + 1) * self.batch_size]
        self.batch_index += 1
        return batch
    
    def next(self):
        """
        Get the next batch of data.
        """
        return self.__next__()

    def reset(self):
        """
        Reset the dataset.
        """
        self.batch_index = 0
        self.shuffle()

    @staticmethod
    def create_dataset(inputs:List[str], batch_size=1, max_len:int=64):
        """
        Create a dataset from a list of input strings.
        """
        tok = tokenizer()
        data = []
        for ins in inputs:
            data.append(create_input_from_string(ins, max_len, tok))
        return Dataset(data, batch_size)

    @staticmethod
    def create_dataset_from_file(filename, tokenizer, max_len, batch_size=1):
        """
        Create a dataset from a file.
        """
        data = []
        with open(filename, 'r') as f:
            content = f.read()
        idx = 0
        input_ids = tokenizer.encode(content)
        while idx < len(input_ids):
            inp = Input(input_ids[idx:idx+max_len+1], shiftable=True)
            # Only add input if it is the right size.
            if len(inp) == max_len:
                data.append(inp)
            idx += max_len
            
        
        return Dataset(data, batch_size)

def create_input_from_string(input_string:str, max_len:int, tokenizer, shiftable=True):
    """
    Create an input from a string.
    """
    input_ids = tokenizer.encode(input_string)
    
    input_ids = preprocess_input(Input(input_ids, shiftable), max_len)
    assert len(input_ids) <= max_len + 1, "Input string is too long."
    return Input(input_ids)

def preprocess_input(input:Input, max_len:int):
    """
    Preprocess the dataset.
    """
    input.value = input.value[:max_len + 1]
    
    return input
