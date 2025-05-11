import torch
import torch.nn as nn
from models.attention_layer import TransformerLayer


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
    invalid_mask = invalid_mask.unsqueeze(1).repeat(1, T, 1)

    temporal_mask = temporal_mask | invalid_mask

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

    pos = x.transpose(1, 2).reshape(N * T, A, 5)[:, :, :2]  # (N*T, A, 2)
    dist = torch.cdist(pos, pos, p=2)  # (N*T, A, A)
    social_mask = dist > config.neighbor_dist  # (N*T, A, A)

    invalid_mask = invalid_entries.transpose(1, 2).reshape(N*T, A)
    invalid_mask = invalid_mask.unsqueeze(1).repeat(1, A, 1)  # (N*T, A, A)

    ego_only_mask = torch.ones_like(dist, dtype=torch.bool)
    ego_only_mask[:, 0, :] = torch.zeros((A), dtype=torch.bool)

    social_mask = social_mask | invalid_mask | ego_only_mask

    # prevent generate NaN during attention
    for i in range(50):
        social_mask[:, i, i] = 0.0
    return social_mask.repeat(config.num_head, 1, 1)


class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_layer = config.num_layer
        self.embedding = nn.Linear(5, config.embed_dim)
        self.encoder_layer = nn.ModuleList([TransformerLayer(config) for _ in range(self.num_layer)])

    def forward(self, x, temporal_mask=None, social_mask=None, distance_bias=None):
        """
        :param x: (N, A, T, 5)
        :param temporal_mask
        :param social_mask
        :return:
        """
        x = self.embedding(x)
        for layer in self.encoder_layer:
            x = layer(x, temporal_mask, social_mask, distance_bias)
        return x


class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.decoder = Transformer(config)
        self.distance_embedding = nn.Embedding(config.num_buckets, config.num_head)
        self.pred_frame = 60
        self.bin_size = config.neighbor_dist / config.num_buckets
        self.mlp = nn.ModuleList([
            nn.Linear(config.embed_dim, config.output_hidden_dim),
            # nn.LeakyReLU(),
            nn.Linear(config.output_hidden_dim, 2 * self.pred_frame)
        ])

    def forward(self, x, invalid_entries):
        """
        x: (N, A, T, 5)
        """
        N, A, T, _ = x.shape
        pos = x.transpose(1, 2).reshape(N * T, A, 5)[:, :, :2]  # (N*T, A, 2)
        dist = torch.cdist(pos, pos, p=2)  # (N*T, A, A)
        bucket_ids = (dist / self.bin_size).long().clamp(max=self.config.num_buckets - 1)
        distance_bias = self.distance_embedding(bucket_ids)  #(N*T, A, A, num_head)
        distance_bias = distance_bias.permute(0, 3, 1, 2).reshape(N*T*self.config.num_head, A, A).to(x.device)

        temporal_mask = get_temporal_mask(x, invalid_entries, self.config)
        social_mask = get_social_mask(x, invalid_entries, self.config)
    
        x = self.decoder(x, temporal_mask, social_mask, distance_bias)  # (N, A, T, D)

        x = x[:, 0]
        for layer in self.mlp:
            x = layer(x)  # (N, T, 2 * pred_frame)

        x = x.reshape(x.shape[0], x.shape[1], -1, 2)  #(N, T, pred_frame, 2)
        return x








