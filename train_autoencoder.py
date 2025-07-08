import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.transforms import v2
import torch.optim as optim
from utils import EarlyStopping

import tqdm
from utils import get_next_dir
import os
import matplotlib.pyplot as plt
from model import AutoEncoder


TRAIN_DIR = get_next_dir(phase='encoder')
os.makedirs(TRAIN_DIR)

# 2. 数据加载（未标注图片）
# transform = v2.Compose([
#     v2.Resize((128, 128)),
#     v2.Grayscale(),  # 转灰度
#     v2.ToTensor(),
# ])
transform = v2.Compose([
    v2.Resize((32, 32)),
    v2.RandomRotation(10),
    v2.RandomHorizontalFlip(),
    v2.RandomAffine(0, translate=(0.1, 0.1)),
    v2.Grayscale(),
    v2.ToTensor(),
])

dataset = ImageFolder(root="data/auto_encoder_imgs", transform=transform)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# 3. 训练 AE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoEncoder().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.L1Loss()

losses = []
best_loss = float('inf')
early_stop = EarlyStopping(delta=0.0001)
for epoch in range(1000):
    model.train()
    total_loss = 0
    for imgs, _ in tqdm.tqdm(dataloader):
        imgs = imgs.to(device)
        outputs = model(imgs)
        loss = loss_fn(outputs, imgs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    total_loss/=len(dataloader)
    if total_loss < best_loss:
        best_loss = total_loss
        torch.save(model.encoder.state_dict(), f"{TRAIN_DIR}/best_encoder.pth")
        torch.save(model, f"{TRAIN_DIR}/best.pt")
    
    early_stop(total_loss, model)
    if early_stop.early_stop:
        break

    losses.append(total_loss)
    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

# 保存训练好的 encoder
torch.save(model, f"{TRAIN_DIR}/last.pt")
torch.save(model.encoder.state_dict(), f"{TRAIN_DIR}/last_encoder.pth")

plt.figure(figsize=(12, 5))
plt.plot(losses, label='Train Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig(f"{TRAIN_DIR}/loss.jpg")
plt.show()
