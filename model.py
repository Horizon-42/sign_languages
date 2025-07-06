import torch.nn as nn
import torch.nn.functional as F

class HandGestureCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(2,2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64*8*8, 128)
        self.fc2 = nn.Linear(128, num_classes)
        

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # [B, 16, 32, 32]
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # [B, 32, 16, 16]
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # [B, 64, 8, 8]

        x = x.view(x.size(0), -1)                       # [B, 64*8*8]
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x