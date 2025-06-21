from models.model import *
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
import numpy as np
import wandb
from datetime import datetime
import argparse
import os
from huggingface_hub import HfApi
from utils.preprocess_data import *


def train(model: nn.Module, train_data: torch.tensor, val_data: torch.tensor, config: argparse.Namespace, device: str, run: wandb.sdk.wandb_run.Run, store_each_epoch: bool = False):
    api = HfApi()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    loss_fn = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=70, eta_min=config.eta_min)
    best_loss = float("inf")

    # create checkpoint path
    os.mkdir(run.name)

    for epoch in range(config.epoch):
        train_pbar = tqdm(DataLoader(train_data, batch_size=config.batch_size, shuffle=True), position=0)
        val_dataloader = DataLoader(val_data, batch_size=config.batch_size)
        train_loss = []
        val_loss = []

        # training
        model.train()
        for traj in train_pbar:
            traj = traj.to(device)
            input_traj = traj[:, :-config.pred_frame, :5]
            N, T, D = traj.shape
            t_idx = torch.arange(T - config.pred_frame, device=traj.device).reshape(-1, 1) + torch.arange(1, config.pred_frame + 1, device=traj.device).reshape(1, -1)
            t_idx = t_idx[None, :, :].repeat(N, 1, 1)
            base_idx = torch.arange(T - config.pred_frame, device=traj.device).reshape(-1, 1) + torch.zeros(config.pred_frame, device=traj.device, dtype=torch.long).reshape(1, -1)
            base_idx = base_idx[None, :, :].repeat(N, 1, 1)
            gt_traj = torch.take_along_dim(traj[:, :, None, :5], t_idx[:, :, :, None], dim=1)
            y_mask = ~((gt_traj[:, :, :, 0] == 0) & (gt_traj[:, :, :, 1] == 0))
            gt_traj[:, :, :, :2] -= torch.take_along_dim(traj[:, :, None, :5], base_idx[:, :, :, None], dim=1)[:, :, :, :2]
            optimizer.zero_grad()

            pred = model(input_traj)  # (N*A, T, pred_frame, 5)
            loss = loss_fn(pred[y_mask], gt_traj[y_mask])
            loss /= N

            loss.backward()
            # torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)
            optimizer.step()

            train_loss.append(loss.item())
            train_pbar.set_description(f"Epoch {epoch}")
            train_pbar.set_postfix({"train loss": loss.item()})

        # check gradient
        for name, param in model.named_parameters():
            if param.grad is not None:
                print(f"{name}: grad mean {param.grad.mean():.5f}, std {param.grad.std():.5f}")

        # validation
        model.eval()
        with torch.no_grad():
            for traj in tqdm(val_dataloader):
                traj = traj.to(device)
                input_traj = traj[:, :50, :5]  # (N, 50, 5)
                gt_traj = traj[:, 50:, :2]  # (N, 60, 2)

                for i in range(60 // config.pred_frame):
                    pred = model(input_traj)[:, -1]  # (N, pred_frame, 5)
                    pred[:, :, :2] += input_traj[:, -1, :2].unsqueeze(1)
                    input_traj = torch.cat([input_traj, pred], dim=1)

                pred_traj = input_traj[:, 50:, :2]
                y_mask = ~((gt_traj[:, :, 0] == 0) & (gt_traj[:, :, 1] == 0))
                loss = loss_fn(pred_traj[y_mask], gt_traj[y_mask])
                val_loss.append(loss.item())

        if epoch < 70:
            scheduler.step()
        mean_val_loss = np.mean(val_loss)
        mean_train_loss = np.mean(train_loss)
        if mean_val_loss < best_loss:
            best_loss = mean_val_loss
            torch.save(model.state_dict(), os.path.join(run.name, f"{run.name}_best.pt"))

        if store_each_epoch:
            if epoch % 4 == 0:
                torch.save(model.state_dict(), os.path.join(run.name, f"{run.name}_{epoch}_{mean_val_loss:.2f}.pt"))

        if epoch % 10 == 0:
            api.upload_folder(
                folder_path=run.name,
                repo_id=config.huggingface_repo,
                path_in_repo=f"{run.name}",
                token=""
            )

        run.log({"Train Loss": mean_train_loss, "Val Loss": mean_val_loss, "Learning Rate": scheduler.get_last_lr()[0]})
        print(f"Epoch {epoch}: Train Loss = {mean_train_loss:.4f}, Val Loss = {np.mean(val_loss):.4f}\n")


def train_w_social_attn(model: nn.Module, train_data: torch.tensor, val_data: torch.tensor, config: argparse.Namespace, device: str, run: wandb.sdk.wandb_run.Run, store_each_epoch: bool = False):
    api = HfApi()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    loss_fn = nn.MSELoss(reduction='sum')
    val_loss_fn = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=120, eta_min=config.eta_min)
    best_loss = float("inf")

    # create checkpoint path
    os.mkdir(run.name)

    for epoch in range(config.epoch):
        train_pbar = tqdm(DataLoader(train_data, batch_size=config.batch_size, shuffle=True), position=0)
        val_dataloader = DataLoader(val_data, batch_size=config.batch_size)
        train_loss = []
        val_loss = []

        # training
        model.train()
        for traj, y_mask, gt_traj, invalid_entries in train_pbar:
            traj, y_mask, gt_traj, invalid_entries = traj.to(device), y_mask.to(device), gt_traj.to(device), invalid_entries.to(device)
            input_traj = traj[:, :, :-60]

            optimizer.zero_grad()

            pred = model(input_traj, invalid_entries)  # (N, pred_frame, 5)

            loss = loss_fn(pred[y_mask], gt_traj[y_mask])
            loss /= pred.shape[0]

            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())
            train_pbar.set_description(f"Epoch {epoch}")
            train_pbar.set_postfix({"train loss": loss.item()})

        # validation
        model.eval()
        with torch.no_grad():
            for traj, y_mask, gt_traj, invalid_entries in tqdm(val_dataloader):
                traj, y_mask, gt_traj, invalid_entries = traj.to(device), y_mask.to(device), gt_traj.to(device), invalid_entries.to(device)
                input_traj = traj[:, :, :-60]

                pred = model(input_traj, invalid_entries)  # (N, 60, 2)
                loss = val_loss_fn(pred[y_mask], gt_traj[y_mask])
                val_loss.append(loss.item())

        if epoch < 70:
            scheduler.step()

        mean_val_loss = np.mean(val_loss)
        mean_train_loss = np.mean(train_loss)
        if mean_val_loss < best_loss:
            best_loss = mean_val_loss
            torch.save(model.state_dict(), os.path.join(run.name, f"{run.name}_best.pt"))

        if store_each_epoch:
            if epoch % 2 == 0:
                torch.save(model.state_dict(), os.path.join(run.name, f"{run.name}_{epoch}_{mean_val_loss:.2f}.pt"))

        if epoch % 4 == 0:
            api.upload_folder(
                folder_path=run.name,
                repo_id=config.huggingface_repo,
                path_in_repo=f"{run.name}",
                token="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"
            )

        run.log({"Train Loss": mean_train_loss, "Val Loss": mean_val_loss, "Learning Rate": scheduler.get_last_lr()[0]})
        print(f"Epoch {epoch}: Train Loss = {mean_train_loss:.4f}, Val Loss = {np.mean(val_loss):.4f}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--embed_dim", default=256, type=int)
    parser.add_argument("--hidden_dim", default=256, type=int)
    parser.add_argument("--dropout", default=0.0, type=float)
    parser.add_argument("--num_head", default=8, type=int)
    parser.add_argument("--num_layer", default=4, type=int)
    parser.add_argument("--output_hidden_dim", default=256, type=int)
    parser.add_argument("--neighbor_dist", default=70, type=int)
    parser.add_argument("--dataset_path", default="dataset")
    parser.add_argument("--huggingface_repo", default="d0703887/CSE251B-Trajectory-Forecasting")
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--lr", default=1e-5, type=float)
    parser.add_argument("--eta_min", default=5e-7, type=float)
    parser.add_argument("--epoch", default=120, type=int)
    parser.add_argument("--num_buckets", default=32, type=int)
    parser.add_argument("--split_val", action="store_true")
    config = parser.parse_args()

    #os.environ['WANDB_MODE'] = 'offline'
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    wandb.login(key="")
    run = wandb.init(
        entity="d0703887",
        project="CSE251b-Trajectory Forecasting",
        name=f"{timestamp}_DecoderOnly",
        config={"embed_dim": config.embed_dim,
                "num_head": config.num_head,
                "hidden_dim": config.hidden_dim,
                "dropout": config.dropout,
                "num_layer": config.num_layer,
                "output_hidden_dim": config.output_hidden_dim,
                "neighbor_dist": config.neighbor_dist,
                "dataset_path": config.dataset_path,
                "huggingface_repo": config.huggingface_repo,

                "batch_size": config.batch_size,
                "lr": config.lr,
                "eta_min": config.eta_min,
                "epoch": config.epoch,
                "num_buckets": config.num_buckets
                },
    )

    model = SequentialEncoder(config)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    split_val = config.split_val
    data = torch.tensor(np.load(os.path.join(config.dataset_path, "train.npz"))["data"], dtype=torch.float32)

    train_data = ArgoverseSocialAttn(data, 'train', split_val, config.dataset_path)
    val_data = ArgoverseSocialAttn(data, 'val', split_val, config.dataset_path)

    train_w_social_attn(model, train_data, val_data, config, device, run, True)

    run.finish()
