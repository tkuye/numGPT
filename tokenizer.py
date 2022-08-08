import json
from typing import List 
import regex as re
import os
import numpy as np

#These are the locations of both the vocab and merges file found from hugging face.

def bytes_to_unicode():
    """
    Returns list of utf-8 byte arrays to unicode strings.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))

def get_pairs(word):
    """
    Return set of symbol pairs in a word.
    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs

class NoVocabError(Exception):
    pass

class Vocab(object):
    def __init__(self, special_tokens=None, vocab_file='vocab.json'):
        self.vocab = {}
        
        self.size = 0
        self.unk_id = None
        self.pad_id = None
        self.bos_id = None
        self.eos_id = None
        if special_tokens is None:
            self.special_tokens = ["<PAD>", "<UNK>", "<SOS>", "<EOS>", "<MASK>"]
        else:
            self.special_tokens = special_tokens
        if os.path.exists(vocab_file):
            
            self.load(vocab_file)
        else:
            raise NoVocabError("Could not find vocab file")
        
    def load(self, filename):
        """
        Load a vocabulary from a file.
        """
        with open(filename, 'r', encoding='utf-8') as f:
            self.vocab = json.load(f)
        self.size = len(self.vocab)

        # We ensure we have our special tokens already added
        if self.special_tokens is not None:
            self.add_special_tokens(self.special_tokens)
        
        self.unk_id = self.vocab.get("<UNK>", None)
        self.pad_id = self.vocab.get("<PAD>", None)
        self.bos_id = self.vocab.get("<BOS>", None)
        self.eos_id = self.vocab.get("<EOS>", None)


    def add_special_tokens(self, tokens):
        for token in tokens:
            self.add(token)

    def __len__(self):
        return self.size
    
    def __contains__(self, word):
        return word in self.vocab
    
    def __getitem__(self, word):
        return self.vocab.get(word, self.unk_id)
    
    def get(self, word, default=None):
        return self.vocab.get(word, default)

    
    def add(self, word):
        """
        Add a word to the vocabulary.
        """
        if word not in self.vocab:
            self.vocab[word] = self.size
            self.size += 1
        
        return self.vocab[word]
    
    def save(self, filename):
        """
        Save the vocabulary to a file.
        """
        with open(filename, 'w') as f:
            json.dump(self.vocab, f)
    
    def __iter__(self):
        return iter(self.vocab)

    def items(self):
        return self.vocab.items()


class Tokenizer(object):
    def __init__(self, vocab, merges_file='merges.txt'):
        self.vocab = vocab
        self.decoder = {v: k for k, v in self.vocab.items()}
        self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        self.byte_encoder = bytes_to_unicode()
        # conver bytes back to unicode string
        with open(merges_file, 'r', encoding='utf-8') as f:
            merges = f.read().split('\n')[1:-1]
        bpe_merges = [tuple(merge.split()) for merge in merges]
        
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
        self.unk_id = self.vocab.get("<UNK>", None)
        self.pad_id = self.vocab.get("<PAD>", None)
        self.bos_id = self.vocab.get("<BOS>", None)
        self.eos_id = self.vocab.get("<EOS>", None)
        self.cache = {}
        self.bpe_cache = {}

    def _tokenize(self, text):
        """
        Tokenize a string using bpe.
        """
        bpe_tokens = []
        for token in re.findall(self.pat, text):
            
            token = "".join(
                self.byte_encoder[b] for b in token.encode("utf-8")
            )
            bpe_tokens.extend(bpe_token for bpe_token in self.bpe(token).split(" "))
        return bpe_tokens
    
    def bpe(self, token):
        """
        BPE tokenization of an input string.
        The algorithm is described in:
        "BPE: Boundary Piece-wise\n 
        Referenced from: https://github.com/huggingface/transformers/blob/b2e4b091f08f1aaf21855d588c6c8d284baba9eb/src/transformers/models/gpt2/tokenization_gpt2.py#L90
        """
        if token in self.cache:
            return self.cache[token]
        word = tuple(token)
        pairs = get_pairs(word)
        
        if not pairs:
            return token # If there are no pairs, just return the token.
        
        while True:
            bigram = min(pairs, key=lambda item: self.bpe_ranks.get(item, float("inf")))
            
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                except ValueError:
                    new_word.extend(word[i:])
                    break
                else:
                    new_word.extend(word[i:j])
                    i = j
                
                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)

        word = " ".join(word)
        self.bpe_cache[token] = word
        return word
                
    def _token_to_id(self, token):
        return self.vocab.get(token, self.vocab.unk_id)

    def _id_to_token(self, idx):
        return self.decoder.get(idx)

    def _tokens_to_ids(self, tokens):
        return [self._token_to_id(token) for token in tokens]
    
    def _ids_to_tokens(self, ids):
        return [self._id_to_token(idx) for idx in ids]
    

    

    def __call__(self, tokens, return_list=False):
        """Converts a string input into its input ids"""
        tokens = self._tokenize(tokens)
        ids = self._tokens_to_ids(tokens)
        if return_list:
            return ids
        else:
            return np.array(ids)

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        text = "".join(tokens)
        text = bytearray([self.byte_decoder[c] for c in text]).decode("utf-8", errors="replace")
        return text    

    def decode(self, ids):
        """Converts a list of ids into a string"""
        tokens = self._ids_to_tokens(ids)
        return self.convert_tokens_to_string(tokens)

    def encode(self, text, return_list=False):
        """Converts a string into a list of ids"""
        tokens = self._tokenize(text)
        ids = self._tokens_to_ids(tokens)
        if return_list:
            return ids
        return np.array(ids)

    def batch_encode(self, texts:List[str], return_list=False):
        """Converts a list of string into a list of ids"""
        ids = [self.encode(text, return_list) for text in texts]
        if return_list:
            return ids
    
        return np.array(ids, dtype=object)
    
    def batch_decode(self, ids:List[List[int]]):
        """Converts a list of ids into a list of strings"""
        texts = [self.decode(ids) for ids in ids]
        return texts

def tokenizer(vocab_file='vocab/vocab.json', merges_file='vocab/merges.txt') -> Tokenizer:
    """
    Build and return tokenizer based on the vocabulary file and merges file.
    """
    vocab = Vocab(vocab_file=vocab_file)
    return Tokenizer(vocab, merges_file)

if __name__ == "__main__":
    ## Test with sample sentences
    vocabulary = Vocab(vocab_file="vocab/vocab.json")
    tokenizer = Tokenizer(vocabulary, merges_file="vocab/merges.txt")
    test_tokens = tokenizer.batch_encode(["Hello, world!", "I'm a sentence.", "I hate black people!"])
    print(test_tokens)
    print(tokenizer.batch_decode(test_tokens))
    

