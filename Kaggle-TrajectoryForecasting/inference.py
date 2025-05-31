from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import numpy as np
from models.model import *
from utils.preprocess_data import *
import argparse
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from utils.dataset_visualization import load_dataset


classes = ['vehicle', 'pedestrian', 'motorcyclist', 'cyclist', 'bus', 'static', 'background', 'construction', 'riderless_bicycle', 'unknown']

# make gif out of a scene.
def make_gif(data_matrix, name='example', pred_traj=None, agent_type=None):
    cmap = plt.cm.get_cmap('viridis', 9)
    fig, ax = plt.subplots(figsize=(10, 10))

    # Function to update plot for each frame
    def update(frame):
        ax.clear()

        # Get data for current timestep
        for i in range(1, data_matrix.shape[0]):
            x = data_matrix[i, frame, 0]
            y = data_matrix[i, frame, 1]
            if x != 0 and y != 0:
                xs = data_matrix[i, :frame + 1, 0]  # Include current frame
                ys = data_matrix[i, :frame + 1, 1]  # Include current frame
                # trim all zeros
                mask = (xs != 0) & (ys != 0)  # Only keep points where both x and y are non-zero
                xs = xs[mask]
                ys = ys[mask]

                # Only plot if we have points to plot
                if len(xs) > 0 and len(ys) > 0:
                    #color = cmap(data_matrix[i, frame, 5])
                    color = cmap(agent_type[i])
                    ax.plot(xs, ys, alpha=0.9, color=color, label=classes[int(agent_type[i].item())])
                    ax.scatter(x, y, s=80, color=color)

                    distance = np.sqrt((x - data_matrix[0, frame, 0])**2 + (y - data_matrix[0, frame, 1])**2)
                    ax.text(x + 1.5, y + 1.5, f'{distance:.1f}', fontsize=8, color='black')

        ax.plot(data_matrix[0, :frame, 0], data_matrix[0, :frame, 1], color='tab:orange', label='Ego Vehicle')
        ax.scatter(data_matrix[0, frame, 0], data_matrix[0, frame, 1], s=80, color='tab:orange')

        if frame >= 50:
            ax.plot(pred_traj[:frame - 50, 0], pred_traj[:frame - 50, 1], color='tab:red', label='Pred Traj')
            ax.scatter(pred_traj[frame - 50, 0], pred_traj[frame - 50, 1], s=80, color='tab:red')
        # Set title with timestep
        ax.set_title(f'Timestep {frame}')

        # Set consistent axis limits
        ax.set_xlim(data_matrix[:, :, 0][data_matrix[:, :, 0] != 0].min() - 10,
                    data_matrix[:, :, 0][data_matrix[:, :, 0] != 0].max() + 10)
        ax.set_ylim(data_matrix[:, :, 1][data_matrix[:, :, 1] != 0].min() - 10,
                    data_matrix[:, :, 1][data_matrix[:, :, 1] != 0].max() + 10)

        # dummy line for each class for the legend
        handles = [ax.plot([], [], label=classes[cls], color=cmap(cls))[0] for cls in range(10)]
        handles.append(plt.Line2D([], [], color='tab:orange', label='Ego Vehicle'))
        handles.append(plt.Line2D([], [], color='tab:red', label='Pred Traj'))
        ax.legend(handles=handles)

        return ax.collections + ax.lines

    # Create animation
    anim = animation.FuncAnimation(fig, update, frames=list(range(0, data_matrix.shape[1], 3)),
                                   interval=100, blit=True)
    #plt.show()
    anim.save(f'visualized_data/{name}.gif', writer='pillow')

def inference(model: nn.Module, test_data, config: argparse.Namespace, device, original_train_data):
    model.to(device)
    loss_fn = nn.MSELoss()
    model.eval()
    losses = []
    cnt = 0
    with torch.no_grad():
        model.eval()
        for traj, y_mask, gt_traj, invalid_entries in tqdm(test_data):
            traj, y_mask, gt_traj, invalid_entries = traj.to(device), y_mask.to(device), gt_traj.to(
                device), invalid_entries.to(device)
            input_traj = traj[:, :, :-60]
            gt_traj = gt_traj[:, -1]

            pred = model(input_traj, invalid_entries)[:, -1, :, :2]  # (N, 60, 2)
            loss = loss_fn(pred[y_mask[:, -1]], gt_traj[y_mask[:, -1]])
            losses.append(loss.item())

            ego_heading = original_train_data[cnt:cnt + config.batch_size, 0, 0, -2].to(device)  # (N)
            cos_h = torch.cos(ego_heading)
            sin_h = torch.sin(ego_heading)
            rot_mats = torch.stack([torch.stack([cos_h, -sin_h], dim=-1),
                                    torch.stack([sin_h, cos_h], dim=-1)], dim=-2).float()  # (N, 2, 2)

            pred = (rot_mats.unsqueeze(1) @ pred.unsqueeze(-1)).squeeze()  # (N, 1, 2, 2) @ (N, 60, 2, 1)
            pred += original_train_data[cnt:cnt + config.batch_size, 0, 0, :2].unsqueeze(1).to(device)
            gt_traj = (rot_mats.unsqueeze(1) @ gt_traj.unsqueeze(-1)).squeeze()
            gt_traj += original_train_data[cnt:cnt + config.batch_size, 0, 0, :2].unsqueeze(1).to(device)
            traj = traj.cpu()
            pred = pred.cpu()
            gt_traj = gt_traj.cpu()
            y_mask = y_mask.cpu()

            for i in range(pred.shape[0]):
                loss = loss_fn(pred[i][y_mask[i, -1]], gt_traj[i][y_mask[i, -1]])

                # if np.rad2deg(abs(original_train_data[cnt, 0, 50, -2] - original_train_data[cnt, 0, -1, -2])) > 30:
                if loss > 10:
                    make_gif(original_train_data[cnt], f"{loss:.2f}_{cnt}", pred[i], original_train_data[cnt, :, 0, -1])

                #make_gif(original_train_data[cnt], str(cnt), pred[i], original_train_data[cnt, :, 0, -1])
                cnt += 1
    print(f"Testing Loss: {np.mean(losses)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--embed_dim", default=256, type=int)
    parser.add_argument("--num_head", default=8, type=int)
    parser.add_argument("--hidden_dim", default=256, type=int)
    parser.add_argument("--dropout", default=0.0, type=float)
    parser.add_argument("--num_layer", default=4, type=int)
    parser.add_argument("--output_hidden_dim", default=256, type=int)
    parser.add_argument("--neighbor_dist", default=70, type=int)
    parser.add_argument("--dataset_path", default="dataset")
    parser.add_argument("--huggingface_repo", default="d0703887/CSE251B-Trajectory-Forecasting")
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--eta_min", default=1e-7, type=float)
    parser.add_argument("--epoch", default=100, type=int)
    parser.add_argument("--num_buckets", default=32, type=int)
    config = parser.parse_args()

    model = Decoder(config)
    data = torch.tensor(np.load("dataset/train.npz")['data'], dtype=torch.float32)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_pbar = tqdm(DataLoader(ArgoverseSocialAttn(data, 'train', False, config.dataset_path), batch_size=config.batch_size, shuffle=False), position=0)
    model.load_state_dict(torch.load("20250530_2358_DecoderOnly/20250530_2358_DecoderOnly_best.pt"))
    inference(model, train_pbar, config, device, data)


