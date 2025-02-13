import copy
import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def layer_init(m, std=np.sqrt(2)):
	"""Custom weight init for Conv2D and Linear layers"""
	if isinstance(m, nn.Linear):
		nn.init.orthogonal_(m.weight.data, std)
		if hasattr(m.bias, 'data'):
			m.bias.data.fill_(0.0)
	elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
		# delta-orthogonal init from https://arxiv.org/pdf/1806.05393.pdf
		assert m.weight.size(2) == m.weight.size(3)
		m.weight.data.fill_(0.0)
		if hasattr(m.bias, 'data'):
			m.bias.data.fill_(0.0)
		mid = m.weight.size(2) // 2
		gain = nn.init.calculate_gain('relu')
		nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)

def get_target(network):
    target = copy.deepcopy(network)
    for param in target.parameters():
        param.requires_grad = False
    return target

def update_target(online_net, target_net, tau=0.5):
    """Update target networks with Polyak averaging. Tau = how much of the online network to use."""
    for target_param, param in zip(target_net.parameters(), online_net.parameters()):
        target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)


class MLP(nn.Module):
    def __init__(
        self,
        hidden_size,
        n_layers,
        inputs=None,
        outputs=None,
    ):
        super().__init__()
        if inputs is None:
            inputs = hidden_size
        if outputs is None:
            outputs = hidden_size
        if n_layers < 2:
            layers = [nn.Linear(inputs, outputs)]
        else:
            layers = [
                nn.Linear(inputs, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.ReLU(),
            ]
            for _ in range(n_layers - 2):
                layers.extend(
                    [
                        nn.Linear(hidden_size, hidden_size),
                        nn.LayerNorm(hidden_size),
                        nn.ReLU(),
                    ]
                )
            layers.append(
                nn.Linear(hidden_size, outputs)
            )
        self.mlp = nn.Sequential(*layers)
        self.apply(layer_init)

    def forward(self, x):
        return self.mlp(x)


class Encoder(nn.Module):
    def __init__(self, base_encoder, hidden_size, variational=False):
        super().__init__()
        self.base_encoder = base_encoder
        self.variational = variational
        self.visual = isinstance(base_encoder, ImageEncoder)
        self.out_dim = hidden_size * 2 if variational else hidden_size
        self.projection = nn.Sequential(
            nn.Linear(base_encoder.out_dim, self.out_dim),
            nn.LayerNorm(self.out_dim),
            nn.Tanh()
    )

    def forward(self, x, detach=False, return_mu_logvar=False):
        x = self.base_encoder(x)
        if detach:
            x = x.detach()
        x = self.projection(x)
        if self.variational:
            mu, logvar = x.chunk(2, dim=-1)
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std
            if return_mu_logvar:
                return z, mu, logvar
            return z
        # non-variational encoder
        if return_mu_logvar:
            return x, x, None
        return x

class ImageEncoder(nn.Module):
    def __init__(self, n_channels, min_res, dropout):
        super().__init__()
        self.n_channels = n_channels
        self.min_res = min_res
        self.encoder = nn.Sequential(
            nn.Conv2d(n_channels, 32, 8, stride=4),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.BatchNorm2d(64),
            nn.Flatten(),
        )
        self.out_dim = 64 * min_res * min_res
        self.apply(layer_init)

    def forward(self, x):
        return self.encoder(x)


class ImageDecoder(nn.Module):
    def __init__(self, n_channels, min_res, hidden_size):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 3 * min_res * min_res),
            nn.Unflatten(1, (3, min_res, min_res)),
            nn.ConvTranspose2d(3, 64, 3, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 32, 4, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, n_channels, 8, stride=4),
            nn.Sigmoid(),
        )
        self.apply(layer_init)

    def forward(self, x):
        return self.decoder(x)


class StateEncoder(nn.Module):
    def __init__(self, state_size, hidden_size, dropout=0):
        super().__init__()
        self.state_size = state_size
        self.out_dim = hidden_size
        self.encoder = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(state_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.apply(layer_init)

    def forward(self, x):
        return self.encoder(x)


class StateDecoder(nn.Module):
    def __init__(self, state_size, hidden_size):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, state_size),
        )
        self.apply(layer_init)

    def forward(self, x):
        return self.decoder(x)


class TransitionModel(nn.Module):
    def __init__(self, hidden_size, action_space, probabilistic, min_sigma, max_sigma):
        super().__init__()
        self.probabilistic = probabilistic
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        self.discrete_action = isinstance(action_space, gym.spaces.Discrete)
        self.actions_dim = (
            action_space.n if self.discrete_action else np.prod(action_space.shape)
        )
        self.transition_model = nn.Sequential(
            nn.Linear(hidden_size + self.actions_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(
                hidden_size,
                hidden_size * 2 if probabilistic else hidden_size,
            ),
        )
        self.apply(layer_init)

    def forward(self, encoded_state, action, return_mu_sigma=False):
        if self.discrete_action:
            action = F.one_hot(action.long(), num_classes=self.actions_dim).float().squeeze()
        x = torch.cat([encoded_state, action], dim=-1)
        x = self.transition_model(x)
        if self.probabilistic:
            mu, log_sigma = x.chunk(2, dim=-1)
            sigma = torch.sigmoid(log_sigma)
            sigma = (
                self.min_sigma + (self.max_sigma - self.min_sigma) * sigma
            )  # rescale to [min_sigma, max_sigma]
            eps = torch.randn_like(sigma)
            z = mu + eps * sigma
            if return_mu_sigma:
                return z, mu, sigma
            return z
        # non-probabilistic transition model
        if return_mu_sigma:
            return x, x, None
        return x
