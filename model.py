import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision.models as models


import torch
import torch.nn as nn
import torch.nn.functional as F

class HandGestureCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)

        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))  # 输出尺寸固定为 1x1
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # [B, 16, H/2, W/2]
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # [B, 32, H/4, W/4]
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # [B, 64, H/8, W/8]
        x = self.pool(F.relu(self.bn4(self.conv4(x))))  # [B, 128, H/16, W/16]
        x = F.relu(self.bn5(self.conv5(x)))             # [B, 256, H/16, W/16]

        x = self.global_pool(x)                         # [B, 256, 1, 1]
        x = torch.flatten(x, 1)                         # [B, 256]
        x = F.relu(self.fc1(x))                         # [B, 256]
        x = self.dropout(x)
        x = self.fc2(x)                                 # [B, num_classes]
        return x


class ResidualBlock(nn.Module):
    """基础残差模块：Conv -> BN -> ReLU -> Conv -> BN + 跳跃连接"""

    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x
        out = F.silu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return F.silu(out)


class EnhancedHandGestureCNN(nn.Module):
    def __init__(self, num_classes, input_size=(64, 64)):
        super().__init__()
        C = 64  # 初始通道数

        self.stage1 = nn.Sequential(
            nn.Conv2d(1, C, kernel_size=3, padding=1),
            nn.BatchNorm2d(C),
            nn.SiLU(),
            nn.MaxPool2d(2, 2),  # -> (C, 32, 32)
        )

        self.stage2 = nn.Sequential(
            nn.Conv2d(C, C*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(C*2),
            nn.SiLU(),
            nn.MaxPool2d(2, 2),  # -> (C*2, 16, 16)
            ResidualBlock(C*2),
        )

        self.stage3 = nn.Sequential(
            nn.Conv2d(C*2, C*4, kernel_size=3, padding=1),
            nn.BatchNorm2d(C*4),
            nn.SiLU(),
            nn.MaxPool2d(2, 2),  # -> (C*4, 8, 8)
            ResidualBlock(C*4),
        )

        self.stage4 = nn.Sequential(
            nn.Conv2d(C*4, C*8, kernel_size=3, padding=1),
            nn.BatchNorm2d(C*8),
            nn.SiLU(),
            ResidualBlock(C*8),
            nn.AdaptiveAvgPool2d((1, 1))  # -> (C*8, 1, 1)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(C*8, 256),
            nn.SiLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.stage1(x)  # [B, 32, 32, 32]
        x = self.stage2(x)  # [B, 64, 16, 16]
        x = self.stage3(x)  # [B, 128, 8, 8]
        x = self.stage4(x)  # [B, 256, 1, 1]
        x = self.classifier(x)
        return x
