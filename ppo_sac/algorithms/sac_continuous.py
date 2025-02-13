import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

import models
import utils
from algorithms import base_sac

LOG_STD_MAX = 2
LOG_STD_MIN = -5


def add_args(parser):
    base_sac.add_args(parser)
    parser.add_argument("--tau", type=float, default=0.005, help="target smoothing coefficient (default: 0.005)")
    parser.add_argument("--q-lr", type=float, default=1e-3, help="the learning rate of the Q network network optimizer")
    parser.add_argument("--policy-frequency", type=int, default=2, help="the frequency of training policy (delayed)")
    parser.add_argument("--target-frequency", type=int, default=1, help="the frequency of updates for the target networks")


class Critic(nn.Module):
    def __init__(self, action_space, encoder: models.Encoder, hidden_size):
        super().__init__()
        self.encoder = encoder
        action_dim = np.prod(action_space.shape)
        self.Q1 = models.MLP(hidden_size, 2, hidden_size + action_dim, 1)
        self.Q2 = models.MLP(hidden_size, 2, hidden_size + action_dim, 1)

    def forward(self, x, a, detach=False):
        x = self.encoder(x, detach)
        x = torch.cat([x, a], 1)
        return self.Q1(x), self.Q2(x)


class Actor(nn.Module):
    def __init__(self, action_space, encoder, hidden_size):
        super().__init__()
        self.encoder = encoder
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc_mean = nn.Linear(hidden_size, np.prod(action_space.shape))
        self.fc_logstd = nn.Linear(hidden_size, np.prod(action_space.shape))
        # action rescaling
        self.register_buffer(
            "action_scale", torch.tensor((action_space.high[0] - action_space.low[0]) / 2.0, dtype=torch.float32).unsqueeze(0)
        )
        self.register_buffer(
            "action_bias", torch.tensor((action_space.high[0] + action_space.low[0]) / 2.0, dtype=torch.float32).unsqueeze(0)
        )
        self.apply(models.layer_init)

    def forward(self, x, detach=False, deterministic=False):
        x = self.encoder(x, detach)
        x = F.relu(self.fc1(x))
        mean = self.fc_mean(x)
        if deterministic:
            return torch.tanh(mean) * self.action_scale + self.action_bias
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean
    
class SAC(base_sac.SAC):
    def __init__(self, envs, args, writer, device):
        super().__init__(envs, args, writer, device)
        if args.autotune:
            self.target_entropy = -torch.prod(torch.Tensor(self.action_space.shape).to(device)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha = self.log_alpha.detach().exp()
            self.a_optimizer = optim.Adam([self.log_alpha], lr=args.alpha_lr, betas=(args.alpha_beta, 0.999))
        else:
            self.alpha = torch.tensor(args.alpha, device=device)

        utils.print_nets([
            ("actor", self.actor),
            ("critic", self.critic),
            ("critic_target", self.critic_target),
        ])
    
    def make_actor(self):
        return Actor(
            self.action_space, 
            self.actor_encoder,
            self.args.hidden_size
        )

    def make_critic(self):
        return Critic(
            self.action_space, 
            self.critic_encoder,
            self.args.hidden_size
        )

    def update_critic(self, data):
        with torch.no_grad():
            next_state_actions, next_state_log_pi, _ = self.actor(data.next_observations)
            qf1_next_target, qf2_next_target = self.critic_target(data.next_observations, next_state_actions)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * self.args.gamma * (min_qf_next_target).view(-1)

        qf1_a_values, qf2_a_values = self.critic(data.observations, data.actions)
        qf1_loss = F.mse_loss(qf1_a_values.view(-1), next_q_value)
        qf2_loss = F.mse_loss(qf2_a_values.view(-1), next_q_value)
        qf_loss = qf1_loss + qf2_loss

        # optimize the model
        self.q_optimizer.zero_grad()
        qf_loss.backward()
        self.q_optimizer.step()

        self.log("losses",
            qf1_values=qf1_a_values.mean().item(), 
            qf2_values=qf2_a_values.mean().item(), 
            qf1_loss=qf1_loss.item(), 
            qf2_loss=qf2_loss.item(), 
            qf_loss=qf_loss.item() / 2.0
        )
    
    def update_actor_and_alpha(self, data):
        pi, log_pi, _ = self.actor(data.observations, detach=True)
        qf1_pi, qf2_pi = self.critic(data.observations, pi, detach=True)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        actor_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.log("losses", actor_loss=actor_loss.item(), alpha=self.alpha)

        if self.args.autotune:
            # dmcontrol-generalization-benchmark/src/algorithms/sac.py uses the same log_pi from the last step
            with torch.no_grad():
                _, log_pi, _ = self.actor(data.observations)
            alpha_loss = (-self.log_alpha.exp() * (log_pi + self.target_entropy)).mean()

            self.a_optimizer.zero_grad()
            alpha_loss.backward()
            self.a_optimizer.step()
            self.alpha.copy_(self.log_alpha.detach().exp())

            self.log("losses", alpha_loss=alpha_loss.item())
