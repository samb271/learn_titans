
from torch.utils.data import DataLoader

from datasets.wikitext import load_wikitext_datasets

def get_loaders(dataset, tokenizer, block_size, batch_size, max_tokens):
        
    # Load the data
    if dataset == "wikitext-103":
        datasets = load_wikitext_datasets(
            tokenizer=tokenizer,
            block_size=block_size,
            max_tokens=max_tokens #262144
        )
    else:
        raise ValueError(f"{dataset} dataset not implemented")
        
    # Create dataloaders
    train_loader = DataLoader(
        datasets["train"], 
        batch_size=batch_size, 
        shuffle=False
    )
    val_loader = DataLoader(
        datasets["val"], 
        batch_size=batch_size, 
        shuffle=False
    )
    test_loader = DataLoader(
        datasets["test"],
        batch_size=batch_size,
        shuffle=False
    )
    
    return train_loader, val_loader, test_loader