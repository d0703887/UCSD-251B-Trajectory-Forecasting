import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from tqdm import tqdm
import os


classes = ['vehicle', 'pedestrian', 'motorcyclist', 'cyclist', 'bus', 'static', 'background', 'construction', 'riderless_bicycle', 'unknown']

# make gif out of a scene.
def make_gif(data_matrix, name='example'):
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
                    color = cmap(data_matrix[i, frame, 5])
                    ax.plot(xs, ys, alpha=0.9, color=color, label=classes[int(data_matrix[i, frame, 5])])
                    ax.scatter(x, y, s=80, color=color)

                    distance = np.sqrt((x - data_matrix[0, frame, 0])**2 + (y - data_matrix[0, frame, 1])**2)
                    ax.text(x + 1.5, y + 1.5, f'{distance:.1f}', fontsize=8, color='black')

        ax.plot(data_matrix[0, :frame, 0], data_matrix[0, :frame, 1], color='tab:orange', label='Ego Vehicle')
        ax.scatter(data_matrix[0, frame, 0], data_matrix[0, frame, 1], s=80, color='tab:orange')
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
        ax.legend(handles=handles)

        return ax.collections + ax.lines

    # Create animation
    anim = animation.FuncAnimation(fig, update, frames=list(range(0, data_matrix.shape[1], 3)),
                                   interval=100, blit=True)
    #plt.show()
    anim.save(f'../visualized_data/trajectory_visualization_{name}.gif', writer='pillow')


def load_dataset(path: str):
    train_data = np.load(os.path.join(path, "train.npz"))["data"]
    test_data = np.load(os.path.join(path, "test_input.npz"))["data"]
    # print("train_data's shape", train_data.shape)
    # print("test_data's shape", test_data.shape)
    return train_data, test_data


if __name__ == "__main__":
    train_data, test_data = load_dataset("../dataset")

    total_frames = 0
    cls_frame = {}
    for i in range(10):
        cls_frame[i] = 0

    for i in range(train_data.shape[0]):
        data_matrix = train_data[i]
        #make_gif(data_matrix, f"index{i}")

        # for j in range(data_matrix.shape[0]):
        #     for frame in range(data_matrix.shape[1]):
        #         if data_matrix[j, frame, 0] != 0 and data_matrix[j, frame, 1] != 0:
        #             cls_frame[data_matrix[j, 0, 5]] += 1
        #             total_frames += 1

        for T in range(50):
            print(f"x={data_matrix[0, T, 0]:>7.3f}, y={data_matrix[0, T, 1]:>7.3f}, vx={data_matrix[0, T, 2]:>5.3f}, vy={data_matrix[0, T, 3]:>5.3f}, heading angle={np.rad2deg(data_matrix[0, T, 4]):.2}")
        exit(0)

    # for cls, frame in cls_frame.items():
    #     print(f"{classes[cls]}: {frame} ({frame / total_frames})")





