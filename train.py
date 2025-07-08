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

from model import HandGestureCNN
from dataset import SignLanguageDataset, read_tensor_dataset, split_tensor_dataset, get_class_names

import tqdm

# --- 1. 配置参数 ---
# 可以根据你的实际情况修改这些参数
NUM_CLASSES = 24                     # 手语手势的类别数量
BATCH_SIZE = 32                      # 每次训练的批量大小
NUM_EPOCHS = 25                      # 训练的总轮数
LEARNING_RATE = 0.001                # 初始学习率
TRAIN_DIR = './runs/train'  # 最佳模型保存路径

if not os.path.exists(TRAIN_DIR):
    os.makedirs(TRAIN_DIR)

# 检查是否有GPU可用
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 读取数据
raw_data = read_tensor_dataset(
    "data/thws-mai-idl-ss-25-sign-language/SignLanguage_kaggle/old_annotated.pth")

label_names = get_class_names(raw_data)
# 对数据进行分割
train_data, val_data, test_data = split_tensor_dataset(raw_data)

# img transform
transform = v2.Resize((64, 64))

train_dataset = SignLanguageDataset(
    train_data, transform=transform)
train_dataloader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True)
train_data_size = len(train_dataset)

val_dataset = SignLanguageDataset(
    val_data, transform=transform)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_data_size = len(val_dataset)

test_dataset = SignLanguageDataset(
    test_data, transform=transform)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
test_data_size = len(test_dataset)


model_ft = HandGestureCNN(num_classes=len(label_names))


# 将模型发送到GPU/CPU
model_ft = model_ft.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer_ft = optim.Adam(model_ft.parameters(), lr=LEARNING_RATE)

# 定义学习率调度器：每7个epoch学习率衰减0.1
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

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
        running_loss = 0.0
        running_corrects = 0
        for inputs, labels in tqdm.tqdm(val_dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)  # 获取预测类别
                loss = criterion(outputs, labels)

            # 统计
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / val_data_size
        epoch_acc = running_corrects.double() / val_data_size

        print(f'Val Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        val_losses.append(epoch_loss)
        val_accs.append(epoch_acc.item())

        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), os.path.join(
                TRAIN_DIR, "best.pt"))

        # save last model
        torch.save(model.state_dict(), os.path.join(
            TRAIN_DIR, "last.pt"))
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
    plt.show()

    return model

# --- 5. 开始训练 ---
if __name__ == '__main__':
    # 运行训练函数
    print("Starting training...")
    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=NUM_EPOCHS)
    print("Training finished.")

    # 可选：在测试集上评估最终模型
    print("\n--- Evaluating on Test Set ---")
    model_ft.eval() # 设置为评估模式
    running_corrects = 0
    with torch.no_grad(): # 不计算梯度
        for inputs, labels in test_dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model_ft(inputs)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)

    test_acc = running_corrects.double() / test_data_size
    print(f'Test Accuracy: {test_acc:.4f}')