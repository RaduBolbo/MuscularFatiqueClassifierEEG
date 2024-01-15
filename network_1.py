from torch import nn
import torch.nn.functional as F


class FatigNet(nn.Module):
    def __init__(self):
        super(FatigNet, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=1)
        
        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Dense (fully connected) layers for binary classification
        self.fc1 = nn.Linear(in_features=1024, out_features=1024)  # Assuming input size of 64x64
        self.fc2 = nn.Linear(in_features=1024, out_features=512)
        self.fc3 = nn.Linear(in_features=512, out_features=128)
        self.fc4 = nn.Linear(in_features=128, out_features=1)  # Binary classification

    def forward(self, x):
        # Applying convolutions and relu activation
        x = x.float()
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = self.pool(F.relu(self.conv4(x)))
        x = F.relu(self.conv5(x))
        x = self.pool(F.relu(self.conv6(x)))

        # Flattening the output for the dense layer
        x = self.global_avg_pool(x) # Adjust size according to your input dimensions
        x = x.view(-1, 1024)

        # Dense layers with relu activation
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        

        # Output layer for binary classification
        x = F.sigmoid(self.fc4(x))
        return x