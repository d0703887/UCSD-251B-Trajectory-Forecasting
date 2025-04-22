from torch.utils.data import Dataset
from utils.dataset_visualization import load_dataset
import torch


class Argoverse(Dataset):
    def __init__(self, mode: str = 'train', split_val: bool = True, dataset_path: str = "dataset"):
        self.data = torch.tensor(load_dataset(dataset_path)[0], dtype=torch.float32)
        if split_val:
            self.data = self.data[:int(len(self.data) * 0.7)] if mode == 'train' else self.data[
                                                                                      int(len(self.data) * 0.7):]

        if mode == 'train':
            self.data = self.data[self.data[:, :, 0, 5] == 0]
        elif mode == 'val' or mode == 'test':
            self.data = self.data[:, 0]

        self.len = len(self.data)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.data[idx]
