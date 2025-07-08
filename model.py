import torch.nn as nn
import torch.nn.functional as F
import torch


class EarlyStopping:
    def __init__(self, patience=5, delta=0.0, save_path=None):
        """
        Args:
            patience (int): 容忍验证集 loss 多轮不下降的次数
            delta (float): 能接受的最小 loss 改善值
            save_path (str): 如果不为 None，会在每次验证 loss 改善时保存模型
        """
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.save_path = save_path

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            if self.save_path:
                torch.save(model.state_dict(), self.save_path)
        elif val_loss > self.best_loss - self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                print(f"Early stopping triggered after {self.counter} epochs.")
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
            if self.save_path:
                torch.save(model.state_dict(), self.save_path)


class HandGestureCNN(nn.Module):
    def __init__(self, num_classes, img_size):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(2,2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)

        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)

        self.dropout = nn.Dropout(0.5)
        # image size after 3 max pool operation
        final_im_size = img_size//(2*2*2*2)
        self.fc1 = nn.Linear(256*final_im_size*final_im_size, 256)
        self.fc2 = nn.Linear(256, num_classes)


    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # [B, 16, 32, 32]
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # [B, 32, 16, 16]
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # [B, 64, 8, 8]
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = F.relu(self.bn5(self.conv5(x)))

        x = x.view(x.size(0), -1)                       # [B, 64*8*8]
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
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
        C = 32  # 初始通道数

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
            nn.Linear(C*8, 128),
            nn.SiLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.stage1(x)  # [B, 32, 32, 32]
        x = self.stage2(x)  # [B, 64, 16, 16]
        x = self.stage3(x)  # [B, 128, 8, 8]
        x = self.stage4(x)  # [B, 256, 1, 1]
        x = self.classifier(x)
        return x
