import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
from tqdm import tqdm
import argparse
import os
import shutil
import datetime
from transformers import AutoTokenizer

from torch.amp import autocast, GradScaler 

from models.transformer import MemoryAugmentedTransformer
from utils.data import get_loaders
from utils.config import load_config

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Set random seed for reproducibility
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


def train(config_path):
    """Train the model using parameters from config file"""
    # Create timestamped run directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("runs", "transformer", timestamp)
    os.makedirs(run_dir, exist_ok=True)
    
    # Load configuration
    config = load_config(config_path)
    
    # Save a copy of the config file in the run directory
    config_filename = os.path.basename(config_path)
    config_save_path = os.path.join(run_dir, config_filename)
    shutil.copy2(config_path, config_save_path)
    print(f"Configuration saved to: {config_save_path}")
    
    # Set model save path
    model_save_path = os.path.join(run_dir, "best_model.pt")
    print(f"Models will be saved to: {model_save_path}")
    
    # Extract configuration values
    model_config = config['model']
    training_config = config['training']
    generation_config = config['generation']
    
    # Set up device
    global device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Set random seed for reproducibility
    torch.manual_seed(training_config['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(training_config['seed'])
        
    tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True)
    vocab_size = tokenizer.vocab_size
    
    train_loader, val_loader, test_loader = get_loaders(
        dataset=training_config["dataset"],
        tokenizer=tokenizer,
        block_size=model_config['block_size'],
        batch_size=training_config['batch_size']
    )
        
    # Initialize the model
    model = MemoryAugmentedTransformer(
        vocab_size=vocab_size,
        d_model=model_config['d_model'],
        num_heads=model_config['num_heads'],
        num_layers=model_config['num_layers'],
        d_ff=model_config['d_ff'],
        max_seq_len=model_config['block_size'],
        memory_depth=model_config['memory_depth'],
        memory_width=model_config['memory_width'],
        dropout=model_config['dropout']
    ).to(device)
    
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    # Initialize optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=training_config['learning_rate'])
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler()
    
    # Training loop
    global_step = 0
    best_val_loss = float('inf')
    
    for epoch in range(training_config['max_epochs']):
        model.train()
        total_loss = 0
        start_time = time.time()
        
        # Add progress bar for each epoch
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{training_config['max_epochs']}")
        
        for batch_idx, (x, y) in progress_bar:
            # Forward pass
            x, y = x.to(device), y.to(device)
            
            with autocast("cuda"):
                logits = model(x)
                logits = logits.reshape(-1, vocab_size)
                y = y.reshape(-1)
                loss = criterion(logits, y)
                    
            total_loss += loss.item()
            
            # Backward pass
            optimizer.zero_grad()
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), training_config['grad_clip'])
            scaler.step(optimizer)
            scaler.update()
            
            global_step += 1
            
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
            
            # Evaluate periodically
            if global_step % training_config['eval_interval'] == 0:
                model.eval()
                val_loss = evaluate(model, val_loader, criterion, vocab_size, device)
                
                # Generate some text
                print(f"\nSamples at step {global_step}:")
                generate_samples(
                    model, 
                    train_loader.dataset, 
                    generation_config['eval_examples'],
                    model_config['block_size'], 
                    device,
                    max_new_tokens=generation_config['max_new_tokens'],
                    temperature=generation_config['temperature'],
                    top_k=generation_config['top_k']
                )
                
                print(f"Epoch {epoch+1}, Step {global_step}, Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}")
                
                # Save the model if it's the best one so far
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), model_save_path)
                    print(f"Model saved to {model_save_path}!")
                
                model.train()
        
        # End of epoch
        elapsed = time.time() - start_time
        print(f"Epoch {epoch+1}/{training_config['max_epochs']} completed in {elapsed:.2f}s, Avg Loss: {total_loss/(batch_idx+1):.4f}")
    
    print("Training completed!")


def evaluate(model, data_loader, criterion, vocab_size, device):
    """Evaluate the model on the validation set"""
    model.eval()
    total_loss = 0
    total_batches = 0
    
    progress_bar = tqdm(data_loader, desc="Evaluating", leave=False)
    
    with torch.no_grad():
        for x, y in progress_bar:
            x, y = x.to(device), y.to(device)
            with autocast("cuda"):
                logits = model(x)
                logits = logits.reshape(-1, vocab_size)
                y = y.reshape(-1)
                loss = criterion(logits, y)
            
            # Calculate loss
            total_loss += loss.item()
            total_batches += 1
    
    return total_loss / total_batches


def generate_samples(model, dataset, num_samples, block_size, device, max_new_tokens=100, temperature=0.8, top_k=40):
    """Generate text samples from the model"""
    model.eval()
    
    for _ in range(num_samples):
        # Start with a random prompt from the dataset
        idx = torch.randint(0, len(dataset) - block_size, (1,))
        context = dataset.data[idx:idx + block_size // 2].unsqueeze(0).to(device)
        
        print(f"Prompt: {dataset.decode(context[0])}")
        
        # Generate text
        with torch.no_grad():
            generated = model.generate(
                context, 
                max_new_tokens=max_new_tokens, 
                temperature=temperature, 
                top_k=top_k
            )
        
        print(f"Generated: {dataset.decode(generated[0])}")
        print("=" * 40)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Train a transformer model with configuration from a YAML file')
    parser.add_argument('--config_path', type=str, required=True, help='Path to the configuration YAML file')
    args = parser.parse_args()
    
    train(args.config_path)
