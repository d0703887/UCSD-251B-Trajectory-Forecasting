from train import Argoverse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from models.model import *
from utils.preprocess_data import *
import argparse

def inference(model: nn.Module, test_data: torch.tensor, config: argparse.Namespace, device):
    model.to(device)
    loss_fn = nn.MSELoss()
    test_pbar = tqdm(DataLoader(test_data, batch_size=config.batch_size), position=0)
    model.eval()
    losses = []
    with torch.no_grad():
        for traj, y_mask in tqdm(test_pbar):
            traj, y_mask = traj.to(device), y_mask.to(device)
            input_traj = traj[:, :50]  # (N, T, 6)
            gt_traj = traj[:, 50:, :2]

            for i in range(60):
                pred = model(input_traj)[:, -1]  # (N, 5)
                pred = torch.cat([pred, torch.zeros(pred.shape[0], 1, device=traj.device)], dim=-1)  # (N, 6)
                input_traj = torch.cat([input_traj, pred[:, None, :]], dim=1)

            pred_traj = input_traj[:, 50:, :2]
            loss = loss_fn(pred_traj[y_mask[:, 50:]], gt_traj[y_mask[:, 50:]])
            pred_traj = pred_traj.cpu()
            gt_traj = gt_traj.cpu()
            y_mask = y_mask.cpu()
            for i in range(10):
                print(pred_traj[y_mask[:, 50:]][i].numpy(), gt_traj[y_mask[:, 50:]][i].numpy())
            print(loss)
            losses.append(loss.item())

    print(f"Testing Loss: {np.mean(losses)}")


if __name__ == "__main__":
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
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--eta_min", default=1e-7, type=float)
    parser.add_argument("--epoch", default=200, type=int)
    parser.add_argument("--use_social_attn", default=False, type=bool)
    config = parser.parse_args()

    model = DecoderOnly(config)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_data = ArgoverseDecoderOnly('train', True, config.dataset_path)
    model.load_state_dict(torch.load("20250420_1645_DecoderOnly_36_1584.62.pt"))
    inference(model, train_data, config, device)


