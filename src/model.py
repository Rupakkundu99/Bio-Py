import torch
import torch.nn as nn
import torch.nn.functional as F

class AlphaFoldCNN(nn.Module):
    """
    A 2D Convolutional Neural Network designed to extract structural features 
    from protein representation matrices.
    
    Input shape: (Batch_Size, 3, Size, Size)
    Output shape: (Batch_Size, 1) - Raw logits representing the thermostability prediction.
    """
    def __init__(self, size: int = 128, dropout_rate: float = 0.3):
        super(AlphaFoldCNN, self).__init__()
        
        # Block 1: Input (3, Size, Size) -> Output (16, Size/2, Size/2)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        
        # Block 2: Input (16, Size/2, Size/2) -> Output (32, Size/4, Size/4)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        
        # Block 3: Input (32, Size/4, Size/4) -> Output (64, Size/8, Size/8)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        
        # Block 4: Input (64, Size/8, Size/8) -> Output (128, Size/16, Size/16)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        
        # Global pooling: collapses spatial dimensions (H, W) to (1, 1)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully connected layers
        self.fc1 = nn.Linear(in_features=128, out_features=64)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(in_features=64, out_features=1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        
        # Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        
        # Block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        
        # Block 4
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        
        # Global Average Pooling
        x = self.global_pool(x)  # Shape: (Batch_Size, 128, 1, 1)
        x = torch.flatten(x, 1)  # Shape: (Batch_Size, 128)
        
        # Fully Connected layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        logits = self.fc2(x)      # Shape: (Batch_Size, 1)
        
        return logits

# Quick verification module
if __name__ == "__main__":
    print("[*] Running model.py verification...")
    # Simulate a batch of 4 proteins of size 128x128
    dummy_input = torch.randn(4, 3, 128, 128)
    model = AlphaFoldCNN(size=128)
    output = model(dummy_input)
    print(f"[+] Input shape: {dummy_input.shape}")
    print(f"[+] Output logits shape: {output.shape} (Expected: (4, 1))")
    
    # Check probabilities
    probs = torch.sigmoid(output)
    print(f"[+] Output probabilities: {probs.squeeze().tolist()}")
