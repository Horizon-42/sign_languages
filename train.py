import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
from torchvision.transforms import v2
import os
import copy
import time
import matplotlib.pyplot as plt
from PIL import Image

from model import *
from dataset import SignLanguageDataset, read_tensor_dataset, split_tensor_dataset, get_class_names, max_channel
from utils import get_next_dir, EarlyStopping, get_last_dir

import tqdm

# --- 1. 配置参数 ---
# 可以根据你的实际情况修改这些参数
NUM_CLASSES = 24                     # 手语手势的类别数量
BATCH_SIZE = 32                      # 每次训练的批量大小
NUM_EPOCHS = 100                      # 训练的总轮数
LEARNING_RATE = 0.001                # 初始学习率
TRAIN_DIR = get_next_dir('runs')
IMAGE_SIZE = 128

if not os.path.exists(TRAIN_DIR):
    os.makedirs(TRAIN_DIR)

print(f"Training in {TRAIN_DIR}")

# 检查是否有GPU可用
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 读取数据
raw_data = read_tensor_dataset(
    "data/thws-mai-idl-ss-25-sign-language/SignLanguage_kaggle/old_annotated.pth")

label_names = get_class_names(raw_data)
# 对数据进行分割
train_data, val_data, test_data = split_tensor_dataset(raw_data, val_rate=0.2)

transform = v2.Compose(
    [
        # v2.Normalize(mean=[0.5339, 0.3282, 0.3282],
        #              std=[0.1378, 0.1967, 0.1967]),
        v2.Lambda(max_channel),  # 输出: [1, H, W]
        v2.Lambda(lambda x: 1-x),
        v2.Normalize([0.3992], [0.1779]),
        # v2.ColorJitter(brightness=0.5, contrast=0.5),
        v2.ToImage(),                                 # 将张量或 PIL 转为 Image
        v2.Resize((IMAGE_SIZE, IMAGE_SIZE)),          # 统一输入尺寸
        # v2.RandomEqualize(p=0.8),                     # 增强边缘/对比度，模拟不同照明条件
        # v2.RandomAffine(
        #     degrees=15,                               # 随机旋转 ±15°
        # ),
        v2.ToDtype(torch.float32, scale=True),        # 转为 [0,1] float32
    ]
)

train_dataset = SignLanguageDataset(
    train_data, transform=transform)
train_dataloader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True)
train_data_size = len(train_dataset)

val_dataset = SignLanguageDataset(
    val_data, transform=transform)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
val_data_size = len(val_dataset)

test_dataset = SignLanguageDataset(
    test_data, transform=transform)
test_dataloader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_data_size = len(test_dataset)


model = EnhancedHandGestureCNN(
    num_classes=len(label_names))
# model = ResNet50ForGesture(num_classes=len(label_names), freeze_backbone=False)

# ENCODE_DIR = get_last_dir(phase='encoder')
# encoder = Encoder()
# encoder.load_state_dict(torch.load(f"{ENCODE_DIR}/best_encoder.pth"))
# model = ClassifierWithEncoder(
#     encoder=encoder, num_classes=NUM_CLASSES, freeze_encoder=True).to(device)

# 将模型发送到GPU/CPU
model = model.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
# already use dropout, didn't need weight_decay
optimizer = optim.Adam(
    model.parameters(), lr=LEARNING_RATE)

# 定义学习率调度器：每7个epoch学习率衰减0.1
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# early stop
early_stopping = EarlyStopping(
    patience=8, delta=0.001, save_path=os.path.join(TRAIN_DIR, "early_stopped.pt"))

# --- 4. 训练函数 ---
def train_model(model, criterion, optimizer, scheduler, num_epochs=NUM_EPOCHS):
    since = time.time() # 记录训练开始时间

    best_model_wts = copy.deepcopy(model.state_dict()) # 复制当前模型状态作为最佳状态
    best_acc = 0.0 # 记录最佳验证准确率

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # ===========================TRAIN==============================
        model.train()

        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in tqdm.tqdm(train_dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)  # get the class for each input
            loss = criterion(outputs, labels)

            # backpropagation
            loss.backward()
            # update parameters
            optimizer.step()
            # statitics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        # update learning rate
        scheduler.step()

        epoch_loss = running_loss / train_data_size
        epoch_acc = running_corrects.double() / train_data_size

        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc.item())

        print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        # ===========================VAL==============================
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        for inputs, labels in tqdm.tqdm(val_dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)  # 获取预测类别
                loss = criterion(outputs, labels)

            # 统计
            val_loss += loss.item() * inputs.size(0)
            val_corrects += torch.sum(preds == labels.data)

        val_epoch_loss = val_loss / val_data_size
        val_epoch_acc = val_corrects.double() / val_data_size

        print(f'Val Loss: {val_epoch_loss:.4f} Acc: {val_epoch_acc:.4f}')

        val_losses.append(val_epoch_loss)
        val_accs.append(val_epoch_acc.item())

        if val_epoch_acc > best_acc:
            best_acc = val_epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), os.path.join(
                TRAIN_DIR, "best.pt"))

        # save last model
        torch.save(model.state_dict(), os.path.join(
            TRAIN_DIR, "last.pt"))

        early_stopping(val_loss=val_epoch_loss, model=model)
        if early_stopping.early_stop:
            break

        print()


    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:.4f}')

    # 加载最佳模型权重
    model.load_state_dict(best_model_wts)

    # 绘制训练曲线
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # 保存图像
    plt.tight_layout()
    plt.savefig(f"{TRAIN_DIR}/accuracy_curve.jpg")

    plt.show()
    return model

# --- 5. 开始训练 ---
if __name__ == '__main__':
    # 运行训练函数
    print("Starting training...")
    model = train_model(model, criterion, optimizer,
                        exp_lr_scheduler, num_epochs=NUM_EPOCHS)
    print("Training finished.")

    # 可选：在测试集上评估最终模型
    print("\n--- Evaluating on Test Set ---")
    model.eval()  # 设置为评估模式
    running_corrects = 0
    with torch.no_grad(): # 不计算梯度
        for inputs, labels in test_dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)

    test_acc = running_corrects.double() / test_data_size
    print(f'Test Accuracy: {test_acc:.4f}')