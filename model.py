import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision.models as models


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


class ResNet50ForGesture(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # 加载预训练的 ResNet50
        self.backbone = models.resnet50(pretrained=True)

        # 修改第一个卷积层为单通道（灰度图）
        self.backbone.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # 替换分类头
        self.backbone.fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)


# 残差块：用于 encoder
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = None
        if downsample or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample:
            identity = self.downsample(x)

        out += identity
        return self.relu(out)


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = ResidualBlock(1, 32, stride=1)
        self.block2 = ResidualBlock(32, 64, stride=2, downsample=True)  # 16x16
        self.block3 = ResidualBlock(64, 128, stride=2, downsample=True)  # 8x8
        self.block4 = ResidualBlock(128, 256, stride=2, downsample=True)  # 4x4

        self.pool = nn.AdaptiveAvgPool2d((1, 1))  # 输出 [B, 256, 1, 1]

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.pool(x)
        return x.view(x.size(0), -1)  # [B, 256]


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(256, 256 * 4 * 4)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=2,
                               padding=1, output_padding=1),
            nn.ReLU(),

            nn.ConvTranspose2d(128, 64, 3, stride=2,
                               padding=1, output_padding=1),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 32, 3, stride=2,
                               padding=1, output_padding=1),
            nn.ReLU(),

            nn.ConvTranspose2d(32, 1, 3, stride=1, padding=1),  # output: 32x32
            nn.Sigmoid()
        )

    def forward(self, x):  # x shape: [B, 256]
        x = self.fc(x).view(-1, 256, 4, 4)
        return self.deconv(x)


class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out


class ClassifierWithEncoder(nn.Module):
    def __init__(self, encoder: nn.Module, num_classes=24, freeze_encoder=True):
        super().__init__()
        self.encoder = encoder

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        features = self.encoder(x)  # [B, 256]
        return self.classifier(features)
