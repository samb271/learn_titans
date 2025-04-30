import os
import torch
from torch.utils.data import Dataset

class WikiTextRawDataset(Dataset):
    """
    Simple dataset wrapper for WikiText corpus that loads raw text
    and doesn't handle tokenization.
    """
    def __init__(self, text, tokenizer, block_size, max_tokens=None):
        """
        Initialize the WikiText dataset.
        
        Args:
            text (str): Raw text from WikiText corpus
            tokenizer: A tokenizer function that converts text to token IDs
            block_size (int): The size of each training sequence
        """
        assert max_tokens > block_size, "Max number of tokens needs to be greater than sequence length"
        
        self.text = text
        self.tokenizer = tokenizer
        self.block_size = block_size

        self.tokens = torch.tensor([], dtype=torch.long)

        print("Starting batched tokenization...")
        # Split the text into manageable chunks (e.g., paragraphs or lines)
        chunks = text.split('\n')  # You can adjust to .split('.') or other heuristics

        for chunk in chunks:
            if self.tokens.size()[0] > max_tokens:
                break
            if not chunk.strip():
                continue  # Skip empty lines
            token_ids = self.tokenizer(chunk, return_tensors="pt").input_ids
            token_ids = token_ids.squeeze(0)
            self.tokens = torch.concat((self.tokens, token_ids))
            
        self.data = self.tokens
        self.vocab_size = getattr(tokenizer, "vocab_size", None)

        print(f"Data size: {len(self.data)} tokens (max: {max_tokens})")
        if self.vocab_size:
            print(f"Vocabulary size: {self.vocab_size}")
        
    def __len__(self):
        return len(self.data) - self.block_size - 1
        
    def __getitem__(self, idx):
        # Get a block of data (x) and the corresponding targets (y)
        # targets are just the input shifted by one position
        end_idx = idx + self.block_size
        block = self.data[idx:end_idx]
        
        # For causal language modeling: input is the block, target is the same block
        x = block[:-1]  # all tokens except last
        y = block[1:]   # all tokens except first (shifted by 1)
        
        return x, y
        
    def decode(self, ids):
        """Convert ids back to text using the tokenizer's decode method if available"""
        if hasattr(self.tokenizer, "decode"):
            return self.tokenizer.decode(ids.tolist())
        return "[Decoding not available]"


def load_wikitext_split(split, tokenizer, block_size, data_dir="./data/wikitext-103", max_tokens=None):
    """
    Load a specific split from WikiText-103 dataset files.
    
    Args:
        split (str): Which dataset split to use ("train", "valid", or "test")
        tokenizer: Tokenizer function or object with a __call__ method that converts text to token IDs
        block_size (int): The size of each training sequence
        data_dir (str): Path to directory containing WikiText-103 files
        
    Returns:
        WikiTextRawDataset: Dataset for the specified split
    """
    # Map split name to file name
    split_file_map = {
        "train": "wiki.train.tokens",
        "val": "wiki.valid.tokens",
        "test": "wiki.test.tokens"
    }
    
    # Check if the split is valid
    if split not in split_file_map:
        raise ValueError(f"Invalid split '{split}'. Must be one of: {list(split_file_map.keys())}")
    
    # Check if the file exists
    file_path = os.path.join(data_dir, split_file_map[split])
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found at {file_path}")
    
    # Read the data file
    print(f"Loading {split} data from {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Create and return dataset
    dataset = WikiTextRawDataset(text, tokenizer, block_size, max_tokens=max_tokens)
    print(f"{split.capitalize()} dataset length: {len(dataset)}")
    
    return dataset


def load_wikitext_datasets(tokenizer, block_size, data_dir="./data/wikitext-103", splits=None, max_tokens=None):
    """
    Load multiple splits from WikiText-103 dataset files.
    
    Args:
        tokenizer: Tokenizer function or object with a __call__ method that converts text to token IDs
        block_size (int): The size of each training sequence
        data_dir (str): Path to directory containing WikiText-103 files
        splits (list): List of splits to load, defaults to ["train", "valid", "test"]
        
    Returns:
        dict: Dictionary mapping split names to their respective datasets
    """
    if splits is None:
        splits = ["train", "val", "test"]
    
    datasets = {}
    for split in splits:
        datasets[split] = load_wikitext_split(split, tokenizer, block_size, data_dir, max_tokens=max_tokens)
    
    return datasets


# Example usage with a simple tokenizer (for demonstration only)
if __name__ == "__main__":
    # This is just a placeholder for demonstration
    class SimpleTokenizer:
        def __init__(self):
            self.vocab = {"<pad>": 0, "<unk>": 1}
            self.vocab_size = len(self.vocab)
            
        def __call__(self, text):
            # Very simple whitespace tokenization
            tokens = text.split()
            # Update vocab
            for token in tokens:
                if token not in self.vocab:
                    self.vocab[token] = len(self.vocab)
            self.vocab_size = len(self.vocab)
            # Convert to IDs
            return [self.vocab.get(token, self.vocab["<unk>"]) for token in tokens]
            
        def decode(self, ids):
            id_to_token = {v: k for k, v in self.vocab.items()}
            return " ".join(id_to_token.get(id, "<unk>") for id in ids)
    
    # Example use
    BLOCK_SIZE = 512
    tokenizer = SimpleTokenizer()
    
    # Load all three splits
    datasets = load_wikitext_datasets(tokenizer, BLOCK_SIZE)
    
    # Or load specific splits as needed
    train_dataset = datasets["train"]
    valid_dataset = datasets["valid"]
    test_dataset = datasets["test"]
    