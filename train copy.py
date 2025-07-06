import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
import os
import copy
import time
import matplotlib.pyplot as plt
from PIL import Image

from model import HandGestureCNN

# --- 1. 配置参数 ---
# 可以根据你的实际情况修改这些参数
DATA_DIR = './hand_gestures_dataset' # 你的数据集根目录
NUM_CLASSES = 24                     # 手语手势的类别数量
BATCH_SIZE = 32                      # 每次训练的批量大小
NUM_EPOCHS = 25                      # 训练的总轮数
LEARNING_RATE = 0.001                # 初始学习率
MODEL_SAVE_PATH = './best_resnet101_model.pth' # 最佳模型保存路径

# 检查是否有GPU可用
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- 2. 数据加载与预处理 ---
# 定义数据转换（数据增强和标准化）
# 训练集需要进行数据增强以提高泛化能力
# 验证集和测试集只进行必要的resize和中心裁剪，然后标准化


def repeat_channels(x):
    return x.repeat(3, 1, 1)

data_transforms = {
    'train': transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Lambda(repeat_channels)
    ]),
    'val': transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Lambda(repeat_channels)
    ]),
    'test': transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Lambda(repeat_channels)
    ]),
}

# 创建ImageFolder数据集
# 你的数据集结构应该如下：
# hand_gestures_dataset/
# ├── train/
# │   ├── class_0/
# │   │   ├── img_001.jpg
# │   │   └── ...
# │   ├── class_1/
# │   │   └── ...
# │   └── ...
# ├── val/
# │   ├── class_0/
# │   │   └── ...
# │   └── ...
# └── test/
#     ├── class_0/
#     │   └── ...
#     └── ...

# 如果你的数据集没有明确的train/val/test文件夹，你需要手动划分
# 这里假设你已经划分好了
image_datasets = {x: datasets.ImageFolder(os.path.join(DATA_DIR, x),
                                          data_transforms[x])
                  for x in ['train', 'val', 'test']}

# 创建数据加载器
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                             batch_size=BATCH_SIZE,
                                             shuffle=True if x == 'train' else False, # 只有训练集需要打乱
                                             num_workers=4) # 可以根据你的CPU核心数调整
               for x in ['train', 'val', 'test']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}
class_names = image_datasets['train'].classes # 获取类别名称
print(f"Dataset sizes: {dataset_sizes}")
print(f"Class names: {class_names}")

# --- 3. 模型构建：使用预训练的ResNet-50 ---
# 加载预训练的ResNet-50模型
# model_ft = models.resnet101(pretrained=True)
model_ft = HandGestureCNN(num_classes=len(class_names))

# # 冻结所有参数
# for param in model_ft.parameters():
#     param.requires_grad = False

# # 替换全连接层 (分类头)
# # ResNet-50的最后一个全连接层是fc
# # 获取fc层的输入特征数
# num_ftrs = model_ft.fc.in_features
# # 替换为新的全连接层，输出维度为你的类别数量
# model_ft.fc = nn.Linear(num_ftrs, NUM_CLASSES)

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

        # 每个epoch都有训练和验证阶段
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # 设置模型为训练模式
            else:
                model.eval()   # 设置模型为评估模式

            running_loss = 0.0
            running_corrects = 0

            # 遍历数据
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # 清零梯度
                optimizer.zero_grad()

                # 前向传播
                # 只有在训练阶段才计算梯度
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1) # 获取预测类别
                    loss = criterion(outputs, labels)

                    # 反向传播 + 优化（仅在训练阶段）
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # 统计
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step() # 更新学习率

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # 记录损失和准确率用于绘图
            if phase == 'train':
                train_losses.append(epoch_loss)
                train_accs.append(epoch_acc.item())
            else:
                val_losses.append(epoch_loss)
                val_accs.append(epoch_acc.item())


            # 深度复制最佳模型
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), MODEL_SAVE_PATH) # 保存最佳模型
                print(f"Saved best model with Acc: {best_acc:.4f} to {MODEL_SAVE_PATH}")


        print() # 每个epoch结束后打印空行

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
        for inputs, labels in dataloaders['test']:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model_ft(inputs)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)

    test_acc = running_corrects.double() / dataset_sizes['test']
    print(f'Test Accuracy: {test_acc:.4f}')