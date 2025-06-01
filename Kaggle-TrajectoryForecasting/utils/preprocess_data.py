from torch.utils.data import Dataset
from utils.dataset_visualization import load_dataset
import torch
import argparse
from tqdm import tqdm


class ArgoverseSocialAttn(Dataset):
    def __init__(self, data, mode: str = 'train', split_val: bool = True, dataset_path: str = "../dataset"):
        if split_val:
            data = data[:int(len(data) * 0.85)] if mode == 'train' else data[int(len(data) * 0.85):]

        y_masks = []
        gt_trajs = []
        datas = []
        invalid_entries = []
        for i in tqdm(range(data.shape[0])):
            cur_data = data[i, :, :, :5]  # (A, T, 5)
            v_linear = torch.sqrt(cur_data[:, :, 2] ** 2 + cur_data[:, :, 3] ** 2)  # (A, T)
            cur_data = torch.cat([cur_data[:, :, :2], v_linear.unsqueeze(-1), cur_data[:, :, 2:]], dim=-1)
            invalid_entries.append((cur_data[:, :50, 0] == 0) & (cur_data[:, :50, 1] == 0))  # (50, 50)


            tmp = cur_data[0, 50:, :2]  #(60, 2)
            y_masks.append(~((tmp[:, 0] == 0) & (tmp[:, 1] == 0)))

            # t_idx = torch.arange(50).reshape(-1, 1) + torch.arange(1, 61).reshape(1, -1)
            # tmp = torch.take_along_dim(cur_data[0, :, None, :2], t_idx[:, :, None], dim=0)  # (50, 60, 5)
            # y_masks.append(~((tmp[:, :, 0] == 0) & (tmp[:, :, 1] == 0)))  # (50, 60)

            # normalize position
            cur_data[:, :, :2] = cur_data[:, :, :2] - cur_data[0, 0, :2]

            # normalize heading angle
            ego_heading = cur_data[0, 0, -1]
            rot_mat = torch.tensor([[torch.cos(-ego_heading), -torch.sin(-ego_heading)], [torch.sin(-ego_heading), torch.cos(-ego_heading)]])

            # rotate heading angle
            cur_data[:, :, -1] = cur_data[:, :, -1] - ego_heading

            # rotate vx, vy
            cur_data[:, :, 3:5] = (rot_mat @ cur_data[:, :, 3:5].unsqueeze(-1)).squeeze()  # (2, 2) @ (A, T, 2, 1)

            # rotate x, y
            cur_data[:, :, :2] = (rot_mat @ cur_data[:, :, :2].unsqueeze(-1)).squeeze()  # (2, 2) @ (A, T, 2, 1)

            datas.append(cur_data)

            gt_trajs.append(cur_data[0, 50:, :2])
            # gt_traj = torch.take_along_dim(cur_data[0, :, None, :2], t_idx[:, :, None], dim=0)
            # gt_trajs.append(gt_traj)

        self.data = datas
        self.y_mask = y_masks
        self.gt_traj = gt_trajs
        self.invalid_entries = invalid_entries
        self.len = len(self.data)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.data[idx], self.y_mask[idx], self.gt_traj[idx], self.invalid_entries[idx]

if __name__ == "__main__":
    from torch.utils.data import DataLoader
    tmp = DataLoader(ArgoverseSocialAttn(), shuffle=True)
    # for data, y_mask, gt in tmp:
    #     print(gt)
    #     exit(0)

