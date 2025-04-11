import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


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
    return temporal_mask.repeat(config['num_head'], 1, 1)


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

    social_mask = dist > config['neighbor_dist']
    #social_mask = social_mask.to(x.device) | invalid_mask
    return social_mask.repeat(config['num_head'], 1, 1)


class SwiGLU(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_dim = config['hidden_dim']
        self.linear1 = nn.Linear(config["embed_dim"], hidden_dim * 2)
        self.linear2 = nn.Linear(hidden_dim, config['embed_dim'])
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.linear1(x)
        x1, x2 = x.chunk(2, dim=-1)
        x = self.act(x1) * x2
        return self.linear2(x)


class EncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config['embed_dim']
        self.num_head = config['num_head']
        self.hidden_dim = config['hidden_dim']
        self.dropout = config['dropout']

        self.temporal_attention = nn.MultiheadAttention(self.embed_dim, self.num_head, batch_first=True)
        self.social_attention = nn.MultiheadAttention(self.embed_dim, self.num_head, batch_first=True)

        self.norm1 = nn.RMSNorm(normalized_shape=self.embed_dim)
        self.norm2 = nn.RMSNorm(normalized_shape=self.embed_dim)
        self.norm3 = nn.RMSNorm(normalized_shape=self.embed_dim)

        self.mlp = SwiGLU(config)

        self.dropout1 = nn.Dropout(self.dropout)
        self.dropout2 = nn.Dropout(self.dropout)
        self.dropout3 = nn.Dropout(self.dropout)

    def forward(self, x, temporal_mask=None, social_mask=None):
        """
        :param x: (N, A, T, D)
        :param temporal_mask:
        :param social_mask:
        :return:
        """
        N, A, T, D = x.shape

        # attend on time
        x = x.reshape(N * A, T, D)
        norm_x = self.norm1(x)
        x = x + self.dropout1(self.temporal_attention(norm_x, norm_x, norm_x, attn_mask=temporal_mask)[0])

        # attend on space
        x = x.reshape(N, A, T, D).transpose(1, 2).reshape(N * T, A, D)
        norm_x = self.norm2(x)
        x = x + self.dropout2(self.social_attention(norm_x, norm_x, norm_x, attn_mask=social_mask)[0])

        x = x.reshape(N, T, A, D).transpose(1, 2)
        x = x + self.dropout3(self.mlp(self.norm3(x)))

        return x


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_layer = config['num_layer']
        self.embedding = nn.Linear(6, config['embed_dim'])
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
        self.encoder = Encoder(config)
        self.mlp = nn.ModuleList([
            nn.Linear(config['embed_dim'], config['output_hidden_dim']),
            nn.LeakyReLU(),
            nn.Linear(config['output_hidden_dim'], 120)
        ])

    def forward(self, x, trainable_mask):
        temporal_mask = get_temporal_mask(x, self.config)
        social_mask = get_social_mask(x, self.config)
        x = self.encoder(x, temporal_mask, social_mask)  # (N, A, T, D)
        x = x[trainable_mask]  # (#trainable samples, T, D)
        x = x[:, -1]
        for layer in self.mlp:
            x = layer(x)
        return x.reshape(x.shape[0], 60, 2)


if __name__ == "__main__":
    config = {"embed_dim": 128,
              "num_head": 8,
              "hidden_dim": 256,
              "dropout": 0.0,
              "num_layer": 4,
              "output_hidden_dim": 256,
              "neighbor_dist": 50
              }
    model = EncoderOnly(config)
    from dataset_visualization import load_dataset
    _, test_data = load_dataset("dataset")
    test_data = torch.tensor(test_data[:2], dtype=torch.float32)

    print(model(test_data))



