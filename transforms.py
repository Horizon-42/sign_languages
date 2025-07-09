from torchvision.transforms import v2
import torch

IMAGE_SIZE = 224

transform = v2.Compose(
    [
        # v2.Lambda(max_channel),  # 输出: [1, H, W]
        # v2.Lambda(lambda tensor: tensor.repeat(3, 1, 1)),  # 输出: [3, H, W]
        v2.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
        v2.ToImage(),                                 # 将张量或 PIL 转为 Image
        v2.Resize((IMAGE_SIZE, IMAGE_SIZE)),          # 统一输入尺寸
        v2.RandomEqualize(p=0.8),                     # 增强边缘/对比度，模拟不同照明条件
        v2.RandomAffine(
            degrees=15,                               # 随机旋转 ±15°
            scale=(0.5, 1.5),                         # 缩放范围
        ),
        v2.ToDtype(torch.float32, scale=True),        # 转为 [0,1] float32
        v2.Normalize(mean=[0.485, 0.456, 0.406],       # ImageNet 的三通道均值
                     std=[0.229, 0.224, 0.225]),
    ]
)