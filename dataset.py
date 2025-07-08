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


def split_tensor_dataset(raw_data: TensorDataset, train_rate=0.7, val_rate=0.15):
    assert train_rate+val_rate <= 0.85  # save space for test data
    total_num = len(raw_data)
    train_num = int(total_num*train_rate)
    val_num = int(total_num*val_rate)
    test_num = total_num-train_num-val_num
    return random_split(raw_data, [train_num, val_num, test_num], generator=torch.Generator().manual_seed(42))


class SignLanguageDataset(Dataset):
    def __init__(self, raw_tensor_dataset: TensorDataset, transform=None, target_transform=None):
        super().__init__()

        self.images, self.labels = raw_tensor_dataset[:]
        # self.transform = transform if transform else v2.Compose(
        #     [v2.ToDtype(torch.float32, scale=True),
        #      v2.Normalize(
        #         mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        self.transform = transform if transform else v2.Grayscale(num_output_channels=1)
        self.target_transform = target_transform if target_transform else Lambda(
            lambda x: x if x <= 8 else x-1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx: int):
        return self.transform(self.images[idx]), self.target_transform(self.labels[idx])
