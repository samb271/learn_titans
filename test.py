import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from copy import deepcopy

class MemoryPerceptron(nn.Module):
    def __init__(self, input_dim, memory_size, key_dim, learning_rate=0.01):
        super(MemoryPerceptron, self).__init__()
        self.keys = nn.Parameter(torch.randn(memory_size, key_dim))
        self.values = nn.Parameter(torch.randn(memory_size, input_dim))
        self.query_proj = nn.Linear(input_dim, key_dim)
        self.learning_rate = learning_rate
        
        # Create separate trainable copies for inner updates
        self.inner_keys = nn.Parameter(self.keys.clone(), requires_grad=False)
        self.inner_values = nn.Parameter(self.values.clone(), requires_grad=False)
        
    def forward(self, x, target=None):
        # Standard forward pass using the main parameters for the outer loop
        query = self.query_proj(x)
        attention = torch.matmul(query, self.keys.t())
        attention = F.softmax(attention, dim=-1)
        predicted_value = torch.matmul(attention, self.values)
        
        # If we have a target (outer layer output), perform inner loop learning
        # on the separate copies without affecting backward graph
        if target is not None:
            with torch.no_grad():
                # Forward pass with inner copies
                inner_attention = torch.matmul(query, self.inner_keys.t())
                inner_attention = F.softmax(inner_attention, dim=-1)
                inner_predicted = torch.matmul(inner_attention, self.inner_values)
                
                # Calculate loss between inner prediction and target
                inner_loss = F.mse_loss(inner_predicted, target.detach())
                
                # Manual gradient calculation (simplified)
                # For keys: gradient is outer product of attention difference and query
                attention_diff = inner_attention.unsqueeze(-1) * (inner_predicted - target.detach()).unsqueeze(1)
                key_grad = torch.einsum('bij,bik->jk', attention_diff, query)
                
                # For values: gradient is based on attention and output difference
                value_grad = torch.einsum('bi,bj->ij', inner_attention, inner_predicted - target.detach())
                
                # Update inner parameters
                self.inner_keys.data -= self.learning_rate * key_grad
                self.inner_values.data -= self.learning_rate * value_grad
                
                # Sync parameters after updating
                self.keys.data.copy_(self.inner_keys.data)
                self.values.data.copy_(self.inner_values.data)
        
        # For the outer loop, we return the combined representation
        return x + predicted_value

class InnerOuterNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, memory_size=10, key_dim=16, inner_lr=0.01):
        super(InnerOuterNetwork, self).__init__()
        
        self.layers = nn.ModuleList()
        self.memory_units = nn.ModuleList()
        
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            self.layers.append(nn.Linear(prev_dim, hidden_dim))
            self.memory_units.append(MemoryPerceptron(hidden_dim, memory_size, key_dim, inner_lr))
            prev_dim = hidden_dim
        
        self.layers.append(nn.Linear(prev_dim, output_dim))
        self.memory_units.append(MemoryPerceptron(output_dim, memory_size, key_dim, inner_lr))
        
    def forward(self, x):
        # First compute all layer outputs
        layer_outputs = []
        current_x = x
        for i, layer in enumerate(self.layers):
            current_x = layer(current_x)
            if i < len(self.layers) - 1:
                current_x = F.relu(current_x)
            layer_outputs.append(current_x)
        
        # Now run the full network with memory units learning
        current_x = x
        for i, (layer, memory) in enumerate(zip(self.layers, self.memory_units)):
            # Outer loop: standard MLP forward
            current_x = layer(current_x)
            if i < len(self.layers) - 1:
                current_x = F.relu(current_x)
            
            # Inner loop: apply memory perceptron with target from first pass
            current_x = memory(current_x, target=layer_outputs[i])
        
        return current_x

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load MNIST data
def load_mnist():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
    
    testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1000, shuffle=False)
    
    return trainloader, testloader

# Training function
def train(model, trainloader, epochs=5):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = inputs.view(inputs.size(0), -1)  # Flatten images
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # This backpropagation affects the outer loop parameters
            # Inner loop parameters are already updated during the forward pass
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            if (i + 1) % 100 == 0:
                print(f'Epoch {epoch+1}, Batch {i+1}, Loss: {running_loss/100:.3f}, Acc: {100.*correct/total:.2f}%')
                running_loss = 0.0
        
        # Evaluate on test set
        acc = evaluate(model, testloader)
        print(f'Epoch {epoch+1} completed. Test accuracy: {acc:.2f}%')

# Evaluation function
def evaluate(model, testloader):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = inputs.view(inputs.size(0), -1)  # Flatten images
            
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    acc = 100. * correct / total
    return acc

# Main execution
if __name__ == "__main__":
    # Load data
    trainloader, testloader = load_mnist()
    
    # Model parameters
    input_dim = 28 * 28  # MNIST images (28x28 pixels)
    hidden_dims = [128, 64]
    output_dim = 10  # 10 classes in MNIST
    memory_size = 20
    key_dim = 32
    inner_learning_rate = 0.01  # Learning rate for the inner loop
    
    # Initialize model
    model = InnerOuterNetwork(
        input_dim=input_dim, 
        hidden_dims=hidden_dims, 
        output_dim=output_dim,
        memory_size=memory_size,
        key_dim=key_dim,
        inner_lr=inner_learning_rate
    )
    
    # Train the model
    train(model, trainloader, epochs=3)
    
    # Final evaluation
    final_acc = evaluate(model, testloader)
    print(f'Final test accuracy: {final_acc:.2f}%')
    
    # Save model
    torch.save(model.state_dict(), 'inner_outer_network_mnist.pth')
    print('Model saved successfully.')