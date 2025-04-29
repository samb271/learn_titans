from datasets.chardataset import CharDataset

def load_data(file_path, seq_len):
    """Load and preprocess the data"""
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Create train and validation splits (90-10 split)
    data_len = len(text)
    train_len = int(data_len * 0.9)
    train_text = text[:train_len]
    val_text = text[train_len:]
    
    print(f"Train text length: {len(train_text)}")
    print(f"Validation text length: {len(val_text)}")
    
    # Create datasets
    train_dataset = CharDataset(train_text, seq_len)
    val_dataset = CharDataset(val_text, seq_len)
    
    return train_dataset, val_dataset