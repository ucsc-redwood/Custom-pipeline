import torch
import torch.nn as nn
import numpy as np

class CustomCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CustomCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Linear(256 * 4 * 4, num_classes)

    def forward(self, x):
        print("Input shape: ", x.shape)
        
        # Reshape the flattened input to [batch_size, channels, height, width]
        x = x.view(x.size(0), 3, 32, 32)
        print("After Reshaping: ", x.shape)
        
        x = self.features[0](x)
        print("After Conv1: ", x.shape)
        x = self.features[1](x)
        x = self.features[2](x)
        print("After MaxPool1: ", x.shape)
        
        x = self.features[3](x)
        print("After Conv2: ", x.shape)
        x = self.features[4](x)
        x = self.features[5](x)
        print("After MaxPool2: ", x.shape)
        
        x = self.features[6](x)
        print("After Conv3: ", x.shape)
        x = self.features[7](x)
        print("After Conv4: ", x.shape)
        x = self.features[8](x)
        print("After Conv5: ", x.shape)
        x = self.features[9](x)
        x = self.features[10](x)
        print("After MaxPool3: ", x.shape)
        
        x = x.view(x.size(0), -1)
        print("After Flattening: ", x.shape)
        
        x = self.classifier(x)
        print("After Linear Layer: ", x.shape)
        
        return x

# Load input data from file
def load_input_data(file_path):
    with open(file_path, 'r') as f:
        data = f.read()
    # Convert string data to a list of floats
    data = np.fromstring(data, sep=' ')
    return torch.tensor(data, dtype=torch.float32)

# Example usage
model = CustomCNN(num_classes=10)
print(model)

# Load the flattened image data from input.txt
input_file = '../images/flattened_dog_dog_1.txt'
input_data = load_input_data(input_file)

# Ensure the input_data is of the correct shape [batch_size, 3*32*32]
if input_data.shape[0] != 3*32*32:
    raise ValueError("Input data must be of shape [3*32*32]")

# Add batch dimension
input_data = input_data.unsqueeze(0)

# Pass the data through the model
output = model(input_data)

