import torch
import torch.nn as nn
from models.attention_layer import *
import torch.nn.functional as F

def get_temporal_mask(x: torch.tensor, invalid_entries, config):
    """
    Generate temporal mask, attending to only previous time step
    :param x: (N, A, T, 6)
    :param invalid_entries: (N, A, T)
    :param config: configuration
    :return: temporal mask
    """
    N, A, T, _ = x.shape

    temporal_mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1).unsqueeze(0).repeat(N * A, 1, 1)  # (N*A, T, T)

    invalid_mask = invalid_entries.reshape(N*A, T)
    invalid_mask = invalid_mask.unsqueeze(1) | invalid_mask.unsqueeze(2)  # (N*A, T, T)
    # invalid_mask = invalid_mask.unsqueeze(1).repeat(1, T, 1)

    temporal_mask = invalid_mask  # | temporal_mask

    # prevent generate NaN during attention
    for i in range(50):
        temporal_mask[:, i, i] = 0.0
    return temporal_mask.repeat(config.num_head, 1, 1)


def get_social_mask(x: torch.tensor, invalid_entries, config):
    """
    Generate social mask, attending to agents within
    :param x: (N, A, T, 6)
    :param invalid_entries: (N, A, T)
    :param config: configuration
    :return: social_mask
    """
    N, A, T, _ = x.shape

    pos = x.transpose(1, 2).reshape(N * T, A, 6)[:, :, :2]  # (N*T, A, 2)
    dist = torch.cdist(pos, pos, p=2)  # (N*T, A, A)
    social_mask = dist > config.neighbor_dist  # (N*T, A, A)

    invalid_mask = invalid_entries.transpose(1, 2).reshape(N*T, A)
    invalid_mask = invalid_mask.unsqueeze(1) | invalid_mask.unsqueeze(2)
    #invalid_mask = invalid_mask.unsqueeze(1).repeat(1, A, 1)  # (N*T, A, A)

    # # only attend ego agent with other agents
    # ego_only_mask = torch.ones_like(dist, dtype=torch.bool)
    # ego_only_mask[:, 0, :] = torch.zeros((A), dtype=torch.bool)

    # only do attention on last time frame
    last_only_mask = torch.ones_like(dist, dtype=torch.bool)
    last_frame_idx = T - 1
    for n in range(N):
        last_only_mask[n * T + last_frame_idx, :, :] = False  # Unmask only last frame

    social_mask = social_mask | invalid_mask | last_only_mask  # | ego_only_mask

    # prevent generate NaN during attention
    for i in range(50):
        social_mask[:, i, i] = 0.0
    return social_mask.repeat(config.num_head, 1, 1)


def get_map_mask(x: torch.tensor, invalid_entries: torch.tensor, config):
    """

    :param x: (N, A, T, 2)
    :param invalid_entries: (N, A, T)
    :param config
    :return:
    """
    N, A, T, _ = x.shape
    flipped_invalid_entires = torch.flip(invalid_entries, dims=[-1])
    num_trailing_trues = torch.argmin(flipped_invalid_entires.int(), dim=-1)  # (N, A)
    max_valid_length = T - torch.min(num_trailing_trues)
    map_points = x[:, :, :max_valid_length].reshape(N, -1, 2)  # (N, M, 2)
    invalid_mask = invalid_entries[:, :, :max_valid_length].reshape(N, -1)  # (N, M)
    invalid_mask = invalid_mask.unsqueeze(1) | invalid_mask.unsqueeze(2)

    # prevent generate NaN during attention
    for i in range(invalid_mask.shape[1]):
        invalid_mask[:, i, i] = 0.0

    return map_points, invalid_mask.repeat(config.num_head, 1, 1)


class Encoder(nn.Module):
    def __init__(self, config, use_rope):
        super().__init__()
        self.config = config
        self.use_rope = use_rope
        self.num_layer = config.num_layer // 2
        self.encoder_layer = nn.ModuleList([EncoderLayer(config, use_rope) for _ in range(self.num_layer)])

    def forward(self, x, attn_mask=None, distance_bias=None):
        """
        :param x
        :param attn_mask
        :param distance_bias
        :return:
        """
        for layer in self.encoder_layer:
            x = layer(x, attn_mask, distance_bias)
        return x


class AgentEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding = nn.Linear(6, config.embed_dim)
        self.temporal_encoder = Encoder(config, True)
        self.social_encoder = Encoder(config, False)
        self.agent_distance_embedding = nn.Embedding(config.num_buckets, config.num_head)
        self.bin_size = config.neighbor_dist / config.num_buckets

    def forward(self, x, invalid_entries):
        """
        x: (N, A, T, 6)
        """
        N, A, T, _ = x.shape
        D = self.config.embed_dim

        # agent features
        pos = x.transpose(1, 2).reshape(N * T, A, 6)[:, :, :2]  # (N*T, A, 2)
        dist = torch.cdist(pos, pos, p=2)  # (N*T, A, A)
        bucket_ids = (dist / self.bin_size).long().clamp(max=self.config.num_buckets - 1)
        distance_bias = self.agent_distance_embedding(bucket_ids)  # (N*T, A, A, num_head)
        distance_bias = distance_bias.permute(0, 3, 1, 2).reshape(N * T * self.config.num_head, A, A).to(x.device)

        temporal_mask = get_temporal_mask(x, invalid_entries, self.config)
        social_mask = get_social_mask(x, invalid_entries, self.config)

        x = self.embedding(x)

        x = x.reshape(N * A, T, D)
        x = self.temporal_encoder(x, temporal_mask)

        x = x.reshape(N, A, T, D).transpose(1, 2).reshape(N * T, A, D)
        x = self.social_encoder(x, social_mask, distance_bias)
        x = x.reshape(N, T, A, D).transpose(1, 2)[:, 0, -1]  # (N, D)
        return x

        # x = x.reshape(N, T, A, D).transpose(1, 2)[:, 0, -1]  # (N, D)
        #
        # for layer in self.mlp:
        #     x = layer(x)  # (N, 2 * pred_frame)
        #
        # x = x.reshape(x.shape[0], -1, 2)  # (N, pred_frame, 2)
        # return x


class MapEncoder(nn.Module):
    """
    Encodes the "Bag of Points" from historical trajectories into a global map context vector.
    This module learns the underlying static structure of the road.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.point_embedding = nn.Linear(2, config.embed_dim)  # Embed (x, y) coordinates

        # A standard Transformer encoder for processing the point cloud
        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(config, use_rope=False) for _ in range(config.map_encoder_layers)])

    def forward(self, x, invalid_entries):
        """
        Args:
            x (torch.Tensor): Shape: (N, A, T, 6)
            invalid_entries (torch.Tensor): Shape: (N, A, T)

        Returns:
            torch.Tensor: A global map context vector for each scene. Shape: (N, D)
        """
        N, A, T, _ = x.shape

        map_points, invalid_mask = get_map_mask(x[:, :, :, :2], invalid_entries, self.config)
        map_points = self.point_embedding(map_points)  # (N, M, D)

        for layer in self.encoder_layers:
            # print(map_points, invalid_mask)
            map_points = layer(map_points, invalid_mask)

        # If no mask, just take the mean
        map_context = map_points.mean(dim=1)  # (N, D)

        return map_context


class Decoder(nn.Module):
    """
    Fuses agent and map embeddings and decodes them into a future trajectory.
    This is designed to be larger and more powerful as requested.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.pred_frame = 60

        # The input dimension is doubled because we concatenate agent and map embeddings
        self.mlp = nn.Sequential(
            nn.Linear(config.embed_dim * 2, config.output_hidden_dim),
            nn.GELU(),
            nn.Linear(config.output_hidden_dim, 2 * self.pred_frame)
        )

    def forward(self, agent_embedding, map_embedding = None):
        """
        Args:
            agent_embedding (torch.Tensor): Ego agent embedding. Shape: (N, D)
            map_embedding (torch.Tensor): Global map context. Shape: (N, D)

        Returns:
            torch.Tensor: Predicted future trajectory. Shape: (N, 60, 2)
        """
        # 1. Fuse the information from the two encoders
        fused_embedding = torch.cat([agent_embedding, map_embedding], dim=1)  # (N, 2*D)

        # 2. Decode the fused embedding into a trajectory
        x = self.mlp(fused_embedding)  # (N, 60 * 2)

        # 3. Reshape to the final trajectory format
        predicted_trajectory = x.view(-1, self.pred_frame, 2)  # (N, 60, 2)

        return predicted_trajectory


class EncoderDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.agent_encoder = AgentEncoder(config)
        self.map_encoder = MapEncoder(config)
        self.decoder = Decoder(config)

    def forward(self, x, invalid_entries):
        """

        :param x:
        :param invalid_entries: (N, A, 50)
        :return:
        """
        N, A, _, _ = x.shape
        agent_embedding = self.agent_encoder(x, invalid_entries)  # (N, D)
        map_embedding = self.map_encoder(x, invalid_entries)  # (N, D)
        predicted_trajectory = self.decoder(agent_embedding, map_embedding)
        return predicted_trajectory


class SequentialEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding = nn.Linear(6, config.embed_dim)
        self.temporal_encoder = Encoder(config, True)
        self.social_encoder = Encoder(config, False)
        self.agent_distance_embedding = nn.Embedding(config.num_buckets, config.num_head)
        self.pred_frame = 60
        self.bin_size = config.neighbor_dist / config.num_buckets
        self.mlp = nn.ModuleList([
            nn.Linear(config.embed_dim, config.output_hidden_dim),
            # nn.LeakyReLU(),
            nn.Linear(config.output_hidden_dim, 2 * self.pred_frame)
        ])

    def forward(self, x, invalid_entries):
        """
        x: (N, A, T, 6)
        """
        N, A, T, _ = x.shape
        D = self.config.embed_dim

        # agent features
        pos = x.transpose(1, 2).reshape(N * T, A, 6)[:, :, :2]  # (N*T, A, 2)
        dist = torch.cdist(pos, pos, p=2)  # (N*T, A, A)
        bucket_ids = (dist / self.bin_size).long().clamp(max=self.config.num_buckets - 1)
        distance_bias = self.agent_distance_embedding(bucket_ids)  # (N*T, A, A, num_head)
        distance_bias = distance_bias.permute(0, 3, 1, 2).reshape(N * T * self.config.num_head, A, A).to(x.device)

        temporal_mask = get_temporal_mask(x, invalid_entries, self.config)
        social_mask = get_social_mask(x, invalid_entries, self.config)

        x = self.embedding(x)

        x = x.reshape(N * A, T, D)
        x = self.temporal_encoder(x, temporal_mask)

        x = x.reshape(N, A, T, D).transpose(1, 2).reshape(N * T, A, D)
        x = self.social_encoder(x, social_mask, distance_bias)

        x = x.reshape(N, T, A, D).transpose(1, 2)[:, 0, -1]  # (N, D)

        for layer in self.mlp:
            x = layer(x)  # (N, 2 * pred_frame)

        x = x.reshape(x.shape[0], -1, 2)  # (N, pred_frame, 2)
        return x


# class EncoderOnly(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.config = config
#         self.agent_encoder = Encoder(config)
#         self.agent_distance_embedding = nn.Embedding(config.num_buckets, config.num_head)
#         self.pred_frame = 60
#         self.bin_size = config.neighbor_dist / config.num_buckets
#         self.mlp = nn.ModuleList([
#             nn.Linear(config.embed_dim, config.output_hidden_dim),
#             # nn.LeakyReLU(),
#             nn.Linear(config.output_hidden_dim, 2 * self.pred_frame)
#         ])
#
#     def forward(self, x, invalid_entries):
#         """
#         x: (N, A, T, 6)
#         """
#         N, A, T, _ = x.shape
#
#         # agent features
#         pos = x.transpose(1, 2).reshape(N * T, A, 6)[:, :, :2]  # (N*T, A, 2)
#         dist = torch.cdist(pos, pos, p=2)  # (N*T, A, A)
#         bucket_ids = (dist / self.bin_size).long().clamp(max=self.config.num_buckets - 1)
#         distance_bias = self.agent_distance_embedding(bucket_ids)  #(N*T, A, A, num_head)
#         distance_bias = distance_bias.permute(0, 3, 1, 2).reshape(N*T*self.config.num_head, A, A).to(x.device)
#
#         temporal_mask = get_temporal_mask(x, invalid_entries, self.config)
#         social_mask = get_social_mask(x, invalid_entries, self.config)
#
#         x = self.agent_encoder(x, temporal_mask, social_mask, distance_bias)  # (N, A, T, D)
#         x = x[:, 0, -1]  # (N, D)
#
#
#         for layer in self.mlp:
#             x = layer(x)  # (N, 2 * pred_frame)
#
#         x = x.reshape(x.shape[0], -1, 2)  # (N, pred_frame, 2)
#         return x








