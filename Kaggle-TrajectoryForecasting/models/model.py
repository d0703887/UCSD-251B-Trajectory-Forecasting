import torch
import torch.nn as nn
from models.attention_layer import EncoderLayer


def get_temporal_mask(x: torch.tensor, config):
    """
    Generate temporal mask, attending to only previous time step
    :param x: (N, A, T, 6)
    :param config: configuration
    :return: temporal mask
    """
    N, A, T, _ = x.shape

    # generate invalid mask to eliminate entries with (x, y) == (0, 0)
    x_reshape = x.reshape(N * A, T, 6)
    invalid_mask = (x_reshape[:, :, 0] == 0) & (x_reshape[:, :, 1] == 0)  # (N*A, T)
    invalid_mask = invalid_mask.unsqueeze(1).repeat(1, T, 1)  # (N*A, T, T)

    temporal_mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1).unsqueeze(0).repeat(N * A, 1, 1)  # (N*A, T, T)
    # temporal_mask = temporal_mask | invalid_mask
    return temporal_mask.repeat(config.num_head, 1, 1)


def get_social_mask(x: torch.tensor, config):
    """
    Generate social mask, attending to agents within
    :param x: (N, A, T, 6)
    :param config: configuration
    :return: social_mask
    """
    N, A, T, _ = x.shape

    # generate invalid mask to eliminate entries with (x, y) == (0, 0)
    x_reshape = x.transpose(1, 2).reshape(N * T, A, 6)
    invalid_mask = (x_reshape[:, :, 0] == 0) & (x_reshape[:, :, 1] == 0)  # (N*T, A)
    invalid_mask = invalid_mask.unsqueeze(1).repeat(1, A, 1)

    pos = x_reshape[:, :, :2]  # (N*T, A, 2)
    dist = torch.sqrt(torch.sum((pos.unsqueeze(2) - pos.unsqueeze(1)) ** 2, dim=-1, keepdim=False))  # (N*T, A, A)

    social_mask = dist > config.neighbor_dist
    #social_mask = social_mask.to(x.device) | invalid_mask
    return social_mask.repeat(config.num_head, 1, 1)


class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_layer = config.num_layer
        self.embedding = nn.Linear(6, config.embed_dim)
        self.encoder_layer = nn.ModuleList([EncoderLayer(config) for _ in range(self.num_layer)])

    def forward(self, x, temporal_mask=None, social_mask=None):
        """
        :param x: (N, A, T, 6)
        :param temporal_mask
        :param social_mask
        :return:
        """
        x = self.embedding(x)
        for layer in self.encoder_layer:
            x = layer(x, temporal_mask, social_mask)
        return x


class EncoderOnly(nn.Module):
    """
    Generate ego agent's next position in an autoregressive way
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder = Transformer(config)
        self.mlp = nn.ModuleList([
            nn.Linear(config.embed_dim, config.output_hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(config.output_hidden_dim, 120)
        ])
        self.use_social_attn = config.use_social_attn

    def forward(self, x, trainable_mask):
        temporal_mask = get_temporal_mask(x, self.config)
        if self.use_social_attn:
            social_mask = get_social_mask(x, self.config)
        else:
            social_mask = None

        x = self.encoder(x, temporal_mask, social_mask)  # (N, A, T, D)
        x = x[trainable_mask]  # (#trainable samples, T, D)
        x = x[:, -1]
        for layer in self.mlp:
            x = layer(x)
        return x.reshape(x.shape[0], 60, 2)


class DecoderOnly(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.decoder = Transformer(config)
        self.use_social_attn = config.use_social_attn
        self.mlp = nn.ModuleList([
            nn.Linear(config.embed_dim, config.output_hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(config.output_hidden_dim, 5)
        ])

    def forward(self, x):
        """
        x: (N*A, T, D)
        """
        x = x.unsqueeze(1)
        temporal_mask = get_temporal_mask(x, self.config)
        if self.use_social_attn:
            social_mask = get_social_mask(x, self.config)
        else:
            social_mask = None

        x = self.decoder(x, temporal_mask, social_mask)  # (N*A, 1, T, D)
        x = x.squeeze()
        for layer in self.mlp:
            x = layer(x)
        return x







