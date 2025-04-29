import torch
from torch.utils.data import Dataset

class CharDataset(Dataset):
    def __init__(self, text, block_size):
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        print(f"Vocabulary size: {self.vocab_size}")
        
        # Character to integer and integer to character mappings
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        
        # Encode the entire text
        self.data = torch.tensor([self.stoi[c] for c in text], dtype=torch.long)
        self.block_size = block_size
        
    def __len__(self):
        return len(self.data) - self.block_size
    
    def __getitem__(self, idx):
        # Get a block of data (x) and the corresponding targets (y)
        # targets are just the input shifted by one position
        x = self.data[idx:idx + self.block_size]
        y = self.data[idx + 1:idx + self.block_size + 1]
        return x, y
    
    def decode(self, ids):
        """Convert ids back to characters"""
        return ''.join([self.itos[int(i)] for i in ids])