import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

class MemoryBloc(nn.Module):
    def __init__(self, depth, width, lr=0.1):
        super().__init__()
        self.lr = torch.tensor(lr)
        self.depth = depth
        self.width = width
        
        # Create weights and corresponding surprise buffers
        for i in range(depth):
            # Initialize weights
            weight = torch.randn(width, width)
            nn.init.xavier_uniform_(weight)
            weight.requires_grad_(True)  # Important for gradient computation
            
            # Register weight and surprise buffer
            self.register_buffer(f"weight_{i}", weight)
            
    def get_named_weights(self) -> List[Tuple[str, torch.Tensor]]:
        return [(name, weight) for name, weight in self.named_buffers()]
    
    def get_weights(self) -> List[torch.Tensor]:
        return [weight for _, weight in self.get_named_weights()]
    
    def zero_grad(self, set_to_none=True):
        """
        Clears gradients for the memory module while ensuring buffers
        do not retain computation history.
        """
        for i in range(self.depth):
            name = f"weight_{i}"
            buffer = getattr(self, name)
            # Detach buffer and require gradients again
            nb = buffer.detach()
            nb.requires_grad_(True)
            self.register_buffer(name, nb)  # Replace with detached buffer
        
        return super().zero_grad(set_to_none)
    
    def update(self, loss, eta=0.9, alpha=0.1):
        
        # Calculate gradients manually
        weights = self.get_weights()
        grads = torch.autograd.grad(loss, weights, create_graph=False)
        
        # Apply updates to weights
        named_weights = self.get_named_weights()
        for (name, weight), grad in zip(named_weights, grads):
            if grad is None:
                continue
                
            # Update weight using gradient descent
            new_weight = weight - self.lr * grad
            
            # Register updated weight as buffer
            self.register_buffer(name, new_weight)
    
    def forward(self, x):
        for i in range(self.depth):
            weight = getattr(self, f"weight_{i}")
            x = torch.matmul(x, weight)
            if i < self.depth - 1:
                x = F.gelu(x)
        return x
        

class MemoryModule(nn.Module):
    def __init__(self, dim, depth=2, heads=1):
        super().__init__()
        self.memory = MemoryBloc(depth, dim)
        self.heads = heads
        
        # Projections for queries, keys, and values
        self.to_queries = nn.Linear(dim, dim, bias=False)
        # self.to_keys = nn.Linear(dim, dim, bias=False)
        # self.to_values = nn.Linear(dim, dim, bias=False)
        
        # Adaptive parameter generators
        self.lr_hyper = 0.01
        
        # State for memory updates
        self.last_momentum = None
        self.stored_memories = None
        
    def retrieve_memories(self, x):
        """Retrieve memories based on input queries"""
        # Project input to queries
        
        with torch.enable_grad():
        
            queries = self.to_queries(x)
            
            # Retrieve memories
            memories = self.memory(queries)
                            
            # Store retrieved memories for reference
            self.stored_memories = memories
            
        # Return detached memories to break computational graph connection
        # This prevents the outer model's backward pass from affecting memory weights
        # and avoids "backward through the graph a second time" errors
        return memories.detach() 
    
    def memorize(self, y):
        """Update memory based on input"""
        
        with torch.enable_grad():
        
            # # Calculate predictions and loss
            loss = F.mse_loss(self.stored_memories, y, reduction='mean')
                    
            self.memory.update(loss)
        
        self.memory.zero_grad()
        
        return loss
    
class TestModel(nn.Module):
    def __init__(self, width, depth):
        super().__init__()
        
        # Create memory module
        self.memory = MemoryModule(dim=width, depth=depth)
        
        # Create a target transformation
        self.Wv = nn.Linear(width, width)
        
    def forward(self, x):
        
        seq_len = x.shape[0]
        
        memories = self.memory.retrieve_memories(x)
        
        memory_enhanced_x = torch.cat([memories, x], dim=0)
        predicted = self.Wv(memory_enhanced_x)
        
        predicted = predicted[-seq_len:, :]
        
        # Inner loop
        mem_loss = self.memory.memorize(predicted)
        
        return predicted, mem_loss
        
    
def test():
        
    # Parameters
    width = 2048
    depth = 4
    num_epochs = 100  # Number of training iterations
    
    model = TestModel(
        width=width,
        depth=depth,
    )
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(params=model.parameters())
    
    # Generate fixed patterns for training
    x_train = torch.randn(10, width)
    y_train = torch.randn(10, width)
    
    # Generate separate validation data
    x_val = torch.randn(10, width)
    y_val = torch.randn(10, width)

    # Track losses
    train_losses = []
    val_losses = []
    memory_losses = []
    
    # Training loop
    for i in range(num_epochs):
        # TRAINING PHASE
        model.train()
        predicted, mem_loss = model(x_train)
        
        # Store memory weights before backward pass
        memory_weights_before = {}
        for name, weight in model.memory.memory.get_named_weights():
            memory_weights_before[name] = weight.clone().detach()
        
        # Outer loop
        outer_loss = criterion(predicted, y_train)
        outer_loss.backward()
        
        # # Check if memory weights changed after backward
        # weights_changed = False
        # for name, weight in model.memory.memory.get_named_weights():
        #     if not torch.allclose(memory_weights_before[name], weight):
        #         weights_changed = True
        #         diff = torch.abs(memory_weights_before[name] - weight).max().item()
        #         print(f"Warning: Weight {name} changed after backward pass! Max diff: {diff}")
        
        # if not weights_changed and i % 10 == 0:
        #     print(f"Iteration {i}: Memory weights unchanged after backward pass (good)")
            
        optimizer.step()
        optimizer.zero_grad()
        
        # Track losses
        train_losses.append(outer_loss.item())
        memory_losses.append(mem_loss.item())
        
        # VALIDATION PHASE
        model.eval()
        with torch.no_grad():
            val_predicted, _ = model(x_val)
            val_loss = criterion(val_predicted, y_val)
            val_losses.append(val_loss.item())
        
        # Print loss and memory usage every few iterations
        if i % 10 == 0:
            
            print(f"Iteration {i}: Train loss = {outer_loss.item():.6f} | "
                  f"Val loss = {val_loss.item():.6f} | "
                  f"Memory loss = {mem_loss.item():.6f}")
        
    # Evaluate final performance
    model.eval()
    with torch.no_grad():
        final_train_pred, _ = model(x_train)
        final_train_loss = criterion(final_train_pred, y_train).item()
        
        final_val_pred, _ = model(x_val)
        final_val_loss = criterion(final_val_pred, y_val).item()
    
    print(f"\nFinal Results:")
    print(f"Training Loss: {final_train_loss:.6f}")
    print(f"Validation Loss: {final_val_loss:.6f}")
    print(f"Generalization Gap: {final_val_loss - final_train_loss:.6f}")


if __name__ == "__main__":
    
    test()