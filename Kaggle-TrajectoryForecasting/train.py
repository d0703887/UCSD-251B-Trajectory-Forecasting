from model import EncoderOnly
from dataset_visualization import load_dataset
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import numpy as np
import wandb
from datetime import datetime


class Argoverse(Dataset):
    def __init__(self, mode='train'):
        if mode == 'train':
            self.data = torch.tensor(load_dataset("dataset")[0], dtype=torch.float32)
            self.data = self.data[:int(len(self.data) * 0.7)]
        elif mode == 'val':
            self.data = torch.tensor(load_dataset("dataset")[0], dtype=torch.float32)
            self.data = self.data[int(len(self.data) * 0.7):]
        elif mode == 'test':
            self.data = torch.tensor(load_dataset("dataset")[1], dtype=torch.float32)

        self.len = len(self.data)
        total_N, A, T, _ = self.data.shape
        self.input_traj = self.data[:, :, :50, :]
        if mode == 'train':
            self.gt_pos = self.data[:, :, 50:, :2]
            self.trainable_mask = self.data[:, :, 0, 5] == 0  # True -> can be used a training sample
            self.y_mask = ~((self.data[:, :, 50:, 0] == 0) & (self.data[:, :, 50:, 1] == 0))  # True -> can be used in loss function
        elif mode == 'val':
            self.gt_pos = self.data[:, :, 50:, :2]
            self.trainable_mask = torch.zeros(self.len, 50, dtype=torch.bool)
            self.trainable_mask[:, 0] = True  # predict only first agent in each scene
            self.y_mask = ~((self.data[:, :, 50:, 0] == 0) & (self.data[:, :, 50:, 1] == 0))
        else:
            self.gt_pos = torch.ones(self.len)
            self.trainable_mask = torch.zeros(self.len, 50, dtype=torch.bool)
            self.trainable_mask[:, 0] = True  # predict only first agent in each scene
            self.y_mask = torch.ones(self.len)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.input_traj[idx], self.gt_pos[idx], self.trainable_mask[idx], self.y_mask[idx]


def train(model: nn.Module, train_data: torch.tensor, val_data: torch.tensor, config: dict, device: str, run: wandb.sdk.wandb_run.Run):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    loss_fn = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 3, gamma=0.5)
    best_loss = float("inf")

    for epoch in range(config['epoch']):
        train_pbar = tqdm(DataLoader(train_data, batch_size=config['batch_size'], shuffle=True), position=0)
        val_dataloader = DataLoader(val_data, batch_size=config['batch_size'])
        train_loss = []
        val_loss = []

        # training
        model.train()
        for input_traj, gt_pos, trainable_mask, y_mask in train_pbar:
            input_traj, gt_pos, trainable_mask, y_mask = input_traj.to(device), gt_pos.to(device), trainable_mask.to(device), y_mask.to(device)
            gt_pos, y_mask = gt_pos[trainable_mask], y_mask[trainable_mask]
            optimizer.zero_grad()

            pred = model(input_traj, trainable_mask)
            loss = loss_fn(pred[y_mask], gt_pos[y_mask])

            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())
            train_pbar.set_description(f"Epoch {epoch}")
            train_pbar.set_postfix({"train loss": loss.item()})

        # validation
        model.eval()
        with torch.no_grad():
            for input_traj, gt_pos, trainable_mask, y_mask in val_dataloader:
                input_traj, gt_pos, trainable_mask, y_mask = input_traj.to(device), gt_pos.to(device), trainable_mask.to(device), y_mask.to(device)
                gt_pos, y_mask = gt_pos[trainable_mask], y_mask[trainable_mask]

                pred = model(input_traj, trainable_mask)
                loss = loss_fn(pred[y_mask], gt_pos[y_mask])
                val_loss.append(loss.item())

        scheduler.step()
        mean_val_loss = np.mean(val_loss)
        mean_train_loss = np.mean(train_loss)
        if mean_val_loss < best_loss:
            best_loss = mean_val_loss
            torch.save(model, f"checkpoints/{run.name}")
        run.log({"Train Loss": mean_train_loss, "Val Loss": mean_val_loss})
        print(f"Epoch {epoch}: Train Loss = {mean_train_loss:.4f}, Val Loss = {np.mean(val_loss):.4f}\n")


if __name__ == "__main__":
    config = {"embed_dim": 168,
              "num_head": 8,
              "hidden_dim": 256,
              "dropout": 0.0,
              "num_layer": 6,
              "output_hidden_dim": 256,
              "neighbor_dist": 50,

              "batch_size": 16,
              "lr": 0.0001,
              "epoch": 20
              }
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    wandb.login(key="8b3e0d688aad58e8826aa06cbd342439d583cdc0")
    run = wandb.init(
        entity="d0703887",
        project="CSE251b-Trajectory Forecasting",
        name=f"{timestamp}_EncoderOnly",
        config=config,
    )

    model = EncoderOnly(config)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_data = Argoverse('train')
    val_data = Argoverse('val')
    train(model, train_data, val_data, config, device, run)
    run.finish()
