# ===== Evaluation: 可视化原图与重建图 =====
import torchvision.utils as vutils
from model import AutoEncoder
from utils import get_last_dir
import torch
import os
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.transforms import v2
import matplotlib.pyplot as plt

MODEL_DIR = get_last_dir(phase="encoder")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model:AutoEncoder = torch.load(os.path.join(MODEL_DIR, "best.pt"))

IMAGE_SIZE = 128

transform = v2.Compose([
    v2.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    v2.Grayscale(),  # 转灰度
    v2.ToTensor(),
])

dataset = ImageFolder(root="data/auto_encoder_imgs", transform=transform)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

model.eval()
sample_imgs = []

todo_count = 0
train_count = 0
# 从 dataloader 中拿前10张图像
with torch.no_grad():
    for imgs, label in dataloader:
        print(label)
        imgs = imgs.to(device)
        outputs = model(imgs)
        sample_imgs = list(zip(imgs.cpu(), outputs.cpu()))
        break  # 只要第一批64张里挑前10张即可

# 保存前10张图（原图 + 重建图）
n_show = 10
fig, axes = plt.subplots(nrows=2, ncols=n_show, figsize=(n_show * 2, 4))
for i in range(n_show):
    # 原图
    axes[0, i].imshow(sample_imgs[i][0].squeeze(), cmap='gray')
    axes[0, i].axis('off')
    axes[0, i].set_title("Original")

    # 重建图
    axes[1, i].imshow(sample_imgs[i][1].squeeze(), cmap='gray')
    axes[1, i].axis('off')
    axes[1, i].set_title("Reconstructed")

plt.suptitle("Original vs Reconstructed (Top: Original, Bottom: Reconstructed)")
plt.tight_layout()
plt.savefig(f"{MODEL_DIR}/recon.jpg")
plt.show()