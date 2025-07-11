import matplotlib.pyplot as plt
from collections.abc import Iterable
import os
from torch.utils.data import TensorDataset

import numpy as np
from torchvision.transforms.functional import to_pil_image
import torch
import re


class EarlyStopping:
    def __init__(self, patience=5, delta=0.0, save_path=None):
        """
        Args:
            patience (int): times you tolerant loss didn't change
            delta (float): lowest change(better) of loss
            save_path (str): save model if it isn't None
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


def plot_dataset(dataset, columns=6, rows=3, cmap=None):
    fig = plt.figure(figsize=(13, 8))

    # ax enables access to manipulate each of subplots
    ax = []
    for i in range(columns * rows):
        img, label = next(dataset) if isinstance(dataset, Iterable) else dataset[i]
        if img.dim() == 4:
            img = img.squeeze(0)
        # create subplot and append to ax
        ax.append(fig.add_subplot(rows, columns, i + 1))
        ax[-1].set_title("Label: " + str(label.item()))
        # rearrage the chanel order
        im = img.permute(1, 2, 0).numpy()
        # im = img.permute(2, 1, 0).numpy
        plt.imshow(im, cmap=cmap)  # show image
    plt.show()  # finally, render the plot


def save_imgs(dataset: TensorDataset, save_dir: str, st_id: int):
    os.makedirs(save_dir, exist_ok=True)  # 创建目录（如果不存在）

    for i in range(len(dataset)):  # unpack TensorDataset elements
        img = dataset[i]
        if img.dim() == 4:
            img = img.squeeze(0)  # 去掉批次维度
        im = img.permute(1, 2, 0)  # CHW -> HWC

        # 如果是单通道或RGB图像，则转为 PIL 保存
        try:
            pil_img = to_pil_image(im)
        except Exception as e:
            # 如果转换失败（如 im 不是 [0,1] 范围或形状错误），尝试标准化处理
            im_np = im.numpy()
            im_np = np.clip(im_np, 0, 1)  # 确保在 [0,1] 范围
            pil_img = to_pil_image(im_np)

        save_path = os.path.join(save_dir, f"{st_id + i}.png")
        pil_img.save(save_path)
    return st_id + len(dataset)


def get_last_dir(root_dir: str = 'runs', phase: str = 'train'):
    paths = os.listdir(root_dir)
    paths = [p for p in paths if phase in p]
    return os.path.join(root_dir, phase+str(len(paths)-1))


def get_next_dir(root_dir: str = 'runs', phase: str = 'train'):
    paths = os.listdir(root_dir)
    paths = [p for p in paths if phase in p]

    return os.path.join(root_dir, phase + str(len(paths)))
