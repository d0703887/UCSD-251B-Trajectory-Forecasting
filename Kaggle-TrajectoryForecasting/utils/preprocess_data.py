from torch.utils.data import Dataset
from utils.dataset_visualization import load_dataset
import torch
import argparse
from tqdm import tqdm

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


class ArgoverseSocialAttn(Dataset):
    def __init__(self, mode: str = 'train', split_val: bool = True, dataset_path: str = "../dataset"):
        data = torch.tensor(load_dataset(dataset_path)[0], dtype=torch.float32)
        if split_val:
            data = data[:int(len(data) * 0.7)] if mode == 'train' else data[int(len(data) * 0.7):]
        #data = data[:10]

        y_masks = []
        gt_trajs = []
        datas = []
        for i in tqdm(range(data.shape[0])):
                cur_data = data[i, :, :, :5]
                t_idx = torch.arange(50).reshape(-1, 1) + torch.arange(1, 61).reshape(1, -1)
                tmp = torch.take_along_dim(cur_data[0, :, None, :], t_idx[:, :, None], dim=0)  # (50, 60, 5)
                y_masks.append(~((tmp[:, :, 0] == 0) & (tmp[:, :, 1] == 0)))  # (50, 60)

                cur_data[:, :, :2] = cur_data[:, :, :2] - cur_data[0, 0, :2]
                datas.append(cur_data)

                gt_traj = torch.take_along_dim(cur_data[0, :, None, :], t_idx[:, :, None], dim=0)
                gt_trajs.append(gt_traj)

        self.data = datas
        self.y_mask = y_masks
        self.gt_traj = gt_trajs
        self.len = len(self.data)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.data[idx], self.y_mask[idx], self.gt_traj[idx]


if __name__ == "__main__":
    ArgoverseSocialAttn()