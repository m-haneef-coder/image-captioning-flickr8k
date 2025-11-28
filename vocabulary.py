"""
Vocabulary builder for image captioning
Handles tokenization and word-to-index mapping
"""

import pickle
from collections import Counter
import os

class Vocabulary:
    """
    Vocabulary class for managing word-to-index mappings
    """
    def __init__(self, min_freq=5):
        """
        Initialize vocabulary with special tokens
        
        Args:
            min_freq: Minimum frequency for a word to be included
        """
        self.word2idx = {}
        self.idx2word = {}
        self. word_freq = Counter()
        self.min_freq = min_freq
        self. idx = 0
        
        # Add special tokens first (ensures consistent indexing)
        self.add_word('<pad>')  # idx 0
        self.add_word('<start>')  # idx 1
        self.add_word('<end>')  # idx 2
        self. add_word('<unk>')  # idx 3
        
    def add_word(self, word):
        """Add a word to vocabulary"""
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1
            
    def build_vocab(self, captions):
        """
        Build vocabulary from list of captions
        
        Args:
            captions: List of caption strings
        """
        print("Building vocabulary...")
        
        # Count word frequencies
        for caption in captions:
            tokens = caption.lower().split()
            for token in tokens:
                self.word_freq[token] += 1
        
        # Add words that meet minimum frequency threshold
        words_added = 0
        words_filtered = 0
        
        for word, freq in self.word_freq. items():
            if freq >= self.min_freq:
                self.add_word(word)
                words_added += 1
            else:
                words_filtered += 1
        
        print(f"✅ Vocabulary built!")
        print(f"   Total unique words: {len(self.word_freq)}")
        print(f"   Words added: {words_added}")
        print(f"   Words filtered (freq < {self.min_freq}): {words_filtered}")
        print(f"   Final vocabulary size: {len(self. word2idx)} (including special tokens)")
        
    def __call__(self, word):
        """
        Get index for a word (returns <unk> index if not found)
        
        Args:
            word: Word to look up
            
        Returns:
            Index of the word
        """
        return self.word2idx.get(word, self.word2idx['<unk>'])
    
    def __len__(self):
        """Return vocabulary size"""
        return len(self.word2idx)
    
    def save(self, filepath):
        """Save vocabulary to file"""
        os.makedirs(os.path. dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f"✅ Vocabulary saved to {filepath}")
            
    @staticmethod
    def load(filepath):
        """Load vocabulary from file"""
        with open(filepath, 'rb') as f:
            vocab = pickle.load(f)
        print(f"✅ Vocabulary loaded from {filepath}")
        print(f"   Vocabulary size: {len(vocab)}")
        return vocab
    
    def get_word(self, idx):
        """Get word from index"""
        return self.idx2word. get(idx, '<unk>')
    
    def encode_caption(self, caption, max_length=30):
        """
        Encode caption to list of indices
        
        Args:
            caption: Caption string
            max_length: Maximum caption length
            
        Returns:
            List of word indices
        """
        tokens = caption.lower().split()
        encoded = [self('<start>')]
        encoded.extend([self(token) for token in tokens])
        encoded.append(self('<end>'))
        
        # Pad or truncate
        if len(encoded) > max_length:
            encoded = encoded[:max_length]
        else:
            encoded += [self('<pad>')] * (max_length - len(encoded))
            
        return encoded
    
    def decode_caption(self, indices, skip_special_tokens=True):
        """
        Decode list of indices to caption string
        
        Args:
            indices: List or tensor of word indices
            skip_special_tokens: Whether to skip <pad>, <start>, <end>
            
        Returns:
            Caption string
        """
        special_tokens = {'<pad>', '<start>', '<end>'} if skip_special_tokens else set()
        
        words = []
        for idx in indices:
            # Handle tensor indices
            if hasattr(idx, 'item'):
                idx = idx.item()
            
            word = self.get_word(idx)
            
            # Stop at <end> token
            if word == '<end>':
                break
                
            if word not in special_tokens:
                words.append(word)
        
        return ' '.join(words)