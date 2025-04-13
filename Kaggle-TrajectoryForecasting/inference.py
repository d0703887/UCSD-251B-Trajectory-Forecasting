from train import Argoverse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def test(model: nn.Module, test_data: torch.tensor, config: dict, device: str):
    model.to(device)
    model.eval()
    loss_fn = nn.MSELoss()
    test_dataloader = DataLoader(test_data, batch_size=config['batch_size'])

    with torch.no_grad():
        for traj in test_dataloader:
            traj = traj.to(device)
            pred


