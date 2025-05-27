

import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicCNN(nn.Module):
    """A basic Convolutional Neural Network (CNN) for FashionMNIST classification.
    
    Architecture:
        - 3 convolutional layers with batch normalization and ReLU activation
        - Max pooling after first two conv layers
        - 2 fully connected layers
        - Dropout for regularization
    
    Input shape: (batch_size, 1, 28, 28)
    Output shape: (batch_size, 10) - logits for 10 fashion item classes
    """
    def __init__(self):
        super(BasicCNN, self).__init__()
        # First convolutional layer: 1 input channel (grayscale) -> 32 feature maps
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)  # Normalize activations for better training
        
        # Second convolutional layer: 32 -> 64 feature maps
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Third convolutional layer: 64 -> 64 feature maps
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        
        # Fully connected layers
        # After 2 max pooling operations (factor of 4 reduction), image size is 7x7
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # Flatten and reduce dimensions
        self.fc2 = nn.Linear(128, 10)  # Final classification layer (10 classes)
        
        # Dropout for regularization (25% dropout rate)
        self.dropout = nn.Dropout(0.25)
    
    def forward(self, x):
        """Forward pass of the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, 28, 28)
            
        Returns:
            torch.Tensor: Output logits of shape (batch_size, 10)
        """
        # First conv block: Conv -> BatchNorm -> ReLU -> MaxPool
        x = F.relu(self.bn1(self.conv1(x)))  # Apply convolution, normalize, and activate
        x = F.max_pool2d(x, 2)  # Reduce spatial dimensions by factor of 2
        
        # Second conv block: similar to first
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        
        # Third conv block: no pooling to preserve spatial information
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Flatten the output for fully connected layers
        x = x.view(x.size(0), -1)  # Reshape to (batch_size, 64 * 7 * 7)
        
        # Fully connected layers with dropout for regularization
        x = F.relu(self.fc1(x))
        x = self.dropout(x)  # Apply dropout to prevent overfitting
        x = self.fc2(x)  # Final classification layer
        
        return x

class ResidualBlock(nn.Module):
    """A basic residual block used in ResNet architectures.
    
    The block implements the residual connection: output = F(x) + x
    where F is a series of convolutions, batch norms, and ReLUs.
    
    Architecture per block:
        - Two 3x3 conv layers with batch normalization
        - Skip connection (identity or 1x1 conv for dimension matching)
        - ReLU activation
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        stride (int): Stride for first convolution and skip connection
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        # First convolution layer with optional striding
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # Second convolution layer always with stride 1
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut connection to match dimensions
        self.shortcut = nn.Sequential()
        # If dimensions change (different channels or stride), add a 1x1 conv
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        """Forward pass of the residual block.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Output after residual connection
        """
        # Main branch
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # Add skip connection
        out += self.shortcut(x)  # Identity or transformed input
        out = F.relu(out)  # Final activation
        return out

class CustomResNet(nn.Module):
    """Custom ResNet architecture adapted for FashionMNIST classification.
    
    A simplified ResNet that works with 28x28 grayscale images.
    The network progressively increases the number of channels while
    reducing spatial dimensions through strided convolutions.
    
    Architecture:
        - Initial conv layer: 1 -> 64 channels
        - 3 residual layers with increasing channels (64->128->256)
        - Global average pooling
        - Final fully connected layer
    
    Args:
        num_classes (int): Number of output classes (default: 10 for FashionMNIST)
    """
    def __init__(self, num_classes=10):
        super(CustomResNet, self).__init__()
        self.in_channels = 64
        
        # Initial convolution layer: grayscale -> 64 channels
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # Residual layers with increasing channels
        self.layer1 = self.make_layer(64, 2, stride=1)   # No spatial reduction
        self.layer2 = self.make_layer(128, 2, stride=2)  # 14x14
        self.layer3 = self.make_layer(256, 2, stride=2)  # 7x7
        
        # Final layers
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))  # Global average pooling
        self.fc = nn.Linear(256, num_classes)  # Final classification
    
    def make_layer(self, out_channels, num_blocks, stride):
        """Creates a layer composed of multiple residual blocks.
        
        Args:
            out_channels (int): Number of output channels
            num_blocks (int): Number of residual blocks in the layer
            stride (int): Stride for first block (for dimensional reduction)
            
        Returns:
            nn.Sequential: A sequence of residual blocks
        """
        # First block may have stride > 1 for dimensional reduction
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(ResidualBlock(self.in_channels, out_channels, stride))
            self.in_channels = out_channels  # Update input channels for next block
        return nn.Sequential(*layers)
    
    def forward(self, x):
        """Forward pass of the ResNet.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, 28, 28)
            
        Returns:
            torch.Tensor: Output logits of shape (batch_size, num_classes)
        """
        # Initial convolution block
        out = F.relu(self.bn1(self.conv1(x)))
        
        # Residual layers
        out = self.layer1(out)  # Keep spatial dimensions
        out = self.layer2(out)  # Reduce to 14x14
        out = self.layer3(out)  # Reduce to 7x7
        
        # Global average pooling and classification
        out = self.avg_pool(out)  # Reduce spatial dimensions to 1x1
        out = out.view(out.size(0), -1)  # Flatten
        out = self.fc(out)  # Final classification
        return out
