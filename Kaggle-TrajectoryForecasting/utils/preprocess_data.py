from torch.utils.data import Dataset
from utils.dataset_visualization import load_dataset
import torch


class Argoverse(Dataset):
    def __init__(self, mode: str = 'train', split_val: bool = True, dataset_path: str = "dataset"):
        self.data = torch.tensor(load_dataset(dataset_path)[0], dtype=torch.float32)
        if split_val:
            self.data = self.data[:int(len(self.data) * 0.7)] if mode == 'train' else self.data[
                                                                                      int(len(self.data) * 0.7):]

        self.len = len(self.data)
        total_N, A, T, _ = self.data.shape
        if mode == 'train':
            self.trainable_mask = self.data[:, :, 0, 5] == 0  # True -> can be used a training sample
            self.y_mask = ~((self.data[:, :, :, 0] == 0) & (self.data[:, :, :, 1] == 0))  # True -> can be used in loss function
        elif mode == 'val':
            self.trainable_mask = torch.zeros(self.len, 50, dtype=torch.bool)
            self.trainable_mask[:, 0] = True  # predict only first agent in each scene
            self.y_mask = ~((self.data[:, :, :, 0] == 0) & (self.data[:, :, :, 1] == 0))
        else:
            self.trainable_mask = torch.zeros(self.len, 50, dtype=torch.bool)
            self.trainable_mask[:, 0] = True  # predict only first agent in each scene
            self.y_mask = torch.ones(self.len)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.data[idx], self.trainable_mask[idx], self.y_mask[idx]
