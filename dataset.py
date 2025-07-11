import os
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataset import TensorDataset
from torch.utils.data import random_split
from torch import Tensor
from torchvision.transforms import Lambda
from torchvision.transforms import v2


def label_to_idx(x: int): return x if x <= 8 else x-1
def idx_to_label(x: int): return x if x <= 8 else x+1


def read_tensor_dataset(file_name: str): return torch.load(
    file_name, weights_only=False)


def get_class_names(data: TensorDataset) -> int:
    _, labels = data[:]
    return sorted(list(set(labels.tolist())))

# Use random split
def split_tensor_dataset(raw_data: TensorDataset, train_rate=0.7, val_rate=0.15):
    total_num = len(raw_data)
    train_num = int(total_num*train_rate)
    val_num = int(total_num*val_rate)
    test_num = total_num-train_num-val_num
    return random_split(raw_data, [train_num, val_num, test_num], generator=torch.Generator().manual_seed(42))


# get max channel of a img
def max_channel(img):
    max_channel_tensor = torch.max(img, dim=0)[0]  # shape: [H, W]
    return max_channel_tensor.unsqueeze(0)  # shape: [1, H, W]



class SignLanguageDataset(Dataset):
    def __init__(self, raw_tensor_dataset: TensorDataset, transform=None, target_transform=None):
        super().__init__()

        self.images, self.labels = raw_tensor_dataset[:]

        # set trans
        if transform:
            self.transform = transform
        else:
            self.transform = lambda x: x

        target_trans = []
        if target_transform:
            target_trans.append(target_transform)
        target_trans.append(Lambda(
            lambda x: x if x <= 8 else x-1))
        self.target_transform = v2.Compose(target_trans)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx: int):
        return self.transform(self.images[idx]), self.target_transform(self.labels[idx])
