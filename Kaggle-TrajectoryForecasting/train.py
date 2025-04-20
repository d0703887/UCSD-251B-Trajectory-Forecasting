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


def decoder_training(model: nn.Module, train_data: torch.tensor, val_data: torch.tensor, config: argparse.Namespace, device: str, run: wandb.sdk.wandb_run.Run, store_each_epoch: bool = False):
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
        for traj, trainable_mask, y_mask in train_pbar:
            traj, trainable_mask, y_mask = traj.to(device), trainable_mask.to(device), y_mask.to(device)
            traj, y_mask = traj[trainable_mask], y_mask[trainable_mask]
            input_traj = traj[:, :-1]
            gt_traj = traj[:, 1:, :5]
            optimizer.zero_grad()

            pred = model(input_traj)  # (N*A, T, 5)
            loss = loss_fn(pred[y_mask[:, 1:]], gt_traj[y_mask[:, 1:]])

            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())
            train_pbar.set_description(f"Epoch {epoch}")
            train_pbar.set_postfix({"train loss": loss.item()})

        # validation
        model.eval()
        with torch.no_grad():
            for traj, trainable_mask, y_mask in tqdm(val_dataloader):
                traj, trainable_mask, y_mask = traj.to(device), trainable_mask.to(device), y_mask.to(device)
                traj, y_mask = traj[trainable_mask], y_mask[trainable_mask]
                input_traj = traj[:, :50]  # (N, T, 6)
                gt_traj = traj[:, 50:, :2]

                for i in range(60):
                    pred = model(input_traj)[:, -1]  # (N, 5)
                    pred = torch.cat([pred, torch.zeros(pred.shape[0], 1, device=traj.device)], dim=-1)  # (N, 6)
                    input_traj = torch.cat([input_traj, pred[:,  None, :]], dim=1)

                pred_traj = input_traj[:, 50:, :2]
                loss = loss_fn(pred_traj[y_mask[:, 50:]], gt_traj[y_mask[:, 50:]])
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
                token="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"
            )

        run.log({"Train Loss": mean_train_loss, "Val Loss": mean_val_loss, "Learning Rate": scheduler.get_last_lr()[0]})
        print(f"Epoch {epoch}: Train Loss = {mean_train_loss:.4f}, Val Loss = {np.mean(val_loss):.4f}\n")


def inference(model: nn.Module, test_data: torch.tensor, config: argparse.Namespace, device):
    model.to(device)
    loss_fn = nn.MSELoss()
    test_pbar = tqdm(DataLoader(test_data, batch_size=config.batch_size), position=0)
    model.eval()
    losses = []
    with torch.no_grad():
        for traj, trainable_mask, y_mask in tqdm(test_pbar):
            traj, trainable_mask, y_mask = traj.to(device), trainable_mask.to(device), y_mask.to(device)
            traj, y_mask = traj[trainable_mask], y_mask[trainable_mask]
            input_traj = traj[:, :50]  # (N, T, 6)
            gt_traj = traj[:, 50:, :2]

            for i in range(60):
                pred = model(input_traj)[:, -1]  # (N, 5)
                pred = torch.cat([pred, torch.zeros(pred.shape[0], 1, device=traj.device)], dim=-1)  # (N, 6)
                input_traj = torch.cat([input_traj, pred[:, None, :]], dim=1)

            pred_traj = input_traj[:, 50:, :2]
            loss = loss_fn(pred_traj[y_mask[:, 50:]], gt_traj[y_mask[:, 50:]])
            losses.append(loss)

    print(f"Testing Loss: {np.mean(losses)}")


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
            torch.save(model.state_dict(), os.path.join(run.name, f"{run.name}_best.pt"))

        if store_each_epoch:
            if epoch % 4 == 0:
                torch.save(model.state_dict(), os.path.join(run.name, f"{run.name}_{epoch}_{mean_val_loss:.2f}.pt"))

        if epoch % 10 == 0:
            api.upload_folder(
                folder_path=run.name,
                repo_id=config.huggingface_repo,
                path_in_repo=f"{run.name}",
                token="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"
            )

        run.log({"Train Loss": mean_train_loss, "Val Loss": mean_val_loss, "Learning Rate": scheduler.get_last_lr()[0]})
        print(f"Epoch {epoch}: Train Loss = {mean_train_loss:.4f}, Val Loss = {np.mean(val_loss):.4f}\n")

if __name__ == "__main__":
    # config = {"embed_dim": 256,
    #           "num_head": 8,
    #           "hidden_dim": 256,
    #           "dropout": 0.0,
    #           "num_layer": 8,
    #           "output_hidden_dim": 256,
    #           "neighbor_dist": 50,
    #           "use_rope": True,
    #           "dataset_path": "dataset",
    #
    #           "batch_size": 8,
    #           "lr": 1e-4,
    #           "eta_min": 1e-7,
    #           "epoch": 200
    #           }

    parser = argparse.ArgumentParser()
    parser.add_argument("--embed_dim", default=256, type=int)
    parser.add_argument("--num_head", default=8, type=int)
    parser.add_argument("--hidden_dim", default=256, type=int)
    parser.add_argument("--dropout", default=0.0, type=float)
    parser.add_argument("--num_layer", default=8, type=int)
    parser.add_argument("--output_hidden_dim", default=256, type=int)
    parser.add_argument("--neighbor_dist", default=50, type=int)
    parser.add_argument("--use_rope", action="store_true")
    parser.add_argument("--dataset_path", default="dataset")
    parser.add_argument("--huggingface_repo", default="d0703887/CSE251B-Trajectory-Forecasting")
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--eta_min", default=1e-7, type=float)
    parser.add_argument("--epoch", default=200, type=int)
    parser.add_argument("--use_social_attn", default=False, type=bool)
    config = parser.parse_args()


    # timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    # wandb.login(key="8b3e0d688aad58e8826aa06cbd342439d583cdc0")
    # run = wandb.init(
    #     entity="d0703887",
    #     project="CSE251b-Trajectory Forecasting",
    #     name=f"{timestamp}_DecoderOnly",
    #     config={"embed_dim": config.embed_dim,
    #             "num_head": config.num_head,
    #             "hidden_dim": config.hidden_dim,
    #             "dropout": config.dropout,
    #             "num_layer": config.num_layer,
    #             "output_hidden_dim": config.output_hidden_dim,
    #             "neighbor_dist": config.neighbor_dist,
    #             "use_rope": config.use_rope,
    #             "dataset_path": config.dataset_path,
    #             "huggingface_repo": config.huggingface_repo,
    #
    #             "batch_size": config.batch_size,
    #             "lr": config.lr,
    #             "eta_min": config.eta_min,
    #             "epoch": config.epoch,
    #             "use_social_attn": config.use_social_attn
    #             },
    # )

    model = DecoderOnly(config)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_data = Argoverse('train', True, config.dataset_path)

    model.load_state_dict(torch.load("20250420_0011_DecoderOnly_best.pt"))
    inference(model, train_data, config, device)
    # val_data = Argoverse('val', True, config.dataset_path)
    # decoder_training(model, train_data, val_data, config, device, run, True)
    # run.finish()

    # model = EncoderOnly(config)
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # train_data = Argoverse('train', False, config.dataset_path)
    # val_data = Argoverse('val', False, config.dataset_path)
    # train(model, train_data, val_data, config, device, run, True)
    # run.finish()
