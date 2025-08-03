# UCSD 251b Trajectory Forecasting
## Intro
Trajectory forecasting, the input is a sequential prediction task. The inputs are $T_{past}$ frames of trajectory of $A$ agents and(or) scene information such as high-definition map, and the output is next $T_{future}$ frames of trajectory of the target agent.

In our case:
$A = 50$
$T_{past} = 50$
$T_{future} = 60$
There is no scene information, only vehicles' trajectory are provided.

At each time step, the state of an agent is represented by 6-dim vector: 
$[x, y, v_{linear}, v_x, v_y, \text{heading angle}]$

## Data Preprocessing
Since dataset's size isn't large enough to train a translation and rotation variant model. We apply the Ego-Centric Alignment data normalization to the dataset.


Specifically, we rotate and translate the coordinate to make ego agent always **at (0, 0) and facing positive x-axis**. This help model focus on relative motion patterns, improve its ability to generalize across varied traffic scenarios.
| Original Coordinate System(orange trajectory denote target agent) | Coordinate System after rotation and translation|
|-|-|
| <img width="350" height="350" alt="5" src="https://github.com/user-attachments/assets/2c8daf06-e913-4ec6-acb6-bf40683e464d" /> | <img width="350" height="350" alt="5_rot" src="https://github.com/user-attachments/assets/70ab59d7-7af8-4316-9d56-f554ae2c4ce6" /> |
| <img width="350" height="350" alt="22" src="https://github.com/user-attachments/assets/a1742290-5d98-4531-8f19-7cc5ed5f072e" /> | <img width="350" height="350" alt="22_rot" src="https://github.com/user-attachments/assets/499109e2-65dd-4639-bd59-1017a6fcd9fa" />



## Method
### Temporal Attention
Temporal Attention captures the evolution of each agents' behavior over time, such as speed pattern and intended destinations. We apply self-attention across the temporal dimensiton without a casual mask.

Furthermore, to inject the concept of time, we apply Rotary Positional Embeddings (RoPE) during the attention.

### Spatial Attention
A large proportion of behavior of vehicle depends on surronding vehicles, pedestrians, bicyclist, and so on. To model the social interactions among agents, we attent each agent to its nearby agents that are within $d_{threshold}$ at the same timestep.

To encode pairwise spatial relationships, we adopt relative positional embeddings similar to T5. Specifically, we compute pariwise Euclidean distances between agents and discretize them into 32 distance bins of size $\frac{d_{threshold}}{32}$. Each bin corresponsing to a learnable embedding that is added to the attention weight to inject distance bias.

### Model Architecture
We implement a **transformer-based encoder** that perform temporal attention and spatial attention inside:
<center><img src="https://hackmd.io/_uploads/By5sM3gHxg.png" /></center>

This encoder predict the future trajectory of target agent in an end-to-end fashion. A lightweight MLP project informative embedding after temporal and spatial attention into future trajectory.

## Results
| Model |  Public Dataset | Private Dataset | Public Leaderboard Ranking | Private Leaderboard Ranking|
|-|-|-|-|-|
|Ours|8.246|8.005|20/69|17/69|

Metric is **Mean Square Error**.




