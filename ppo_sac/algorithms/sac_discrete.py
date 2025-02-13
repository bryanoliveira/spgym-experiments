import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.distributions.categorical import Categorical
import models
import utils
from algorithms import base_sac


def add_args(parser):
    base_sac.add_args(parser)
    parser.add_argument("--tau", type=float, default=0.005, help="target smoothing coefficient - how much of the online network to use (default: 1)")
    parser.add_argument("--target-frequency", type=int, default=1, help="the frequency of updates for the target networks")
    parser.add_argument("--q-lr", type=float, default=3e-4, help="the learning rate of the Q network network optimizer")
    parser.add_argument("--policy-frequency", type=int, default=2, help="the frequency of training updates")
    parser.add_argument("--policy-lr", type=float, default=3e-4, help="the learning rate of the policy network optimizer")
    parser.add_argument("--target-entropy-scale", type=float, default=0.89, help="coefficient for scaling the autotune entropy target")
    parser.add_argument("--autotune", type=bool, default=False, action=argparse.BooleanOptionalAction, help="automatic tuning of the entropy coefficient")
    parser.add_argument("--alpha", type=float, default=0.05, help="Entropy regularization coefficient.")
    parser.add_argument("--alpha-lr", type=float, default=3e-4, help="the learning rate of the entropy coefficient optimizer")
    parser.add_argument("--alpha-beta", type=float, default=0.5, help="the beta of the entropy coefficient optimizer")


class Critic(nn.Module):
    def __init__(self, actions_dim, encoder: models.Encoder, hidden_size: int):
        super().__init__()
        self.encoder = encoder
        self.Q1 = models.MLP(hidden_size, 2, outputs=actions_dim)
        self.Q2 = models.MLP(hidden_size, 2, outputs=actions_dim)

    def forward(self, x, detach=False):
        x = self.encoder(x, detach)
        return self.Q1(x), self.Q2(x)


class Actor(nn.Module):
    def __init__(self, actions_dim, encoder: models.Encoder, hidden_size: int):
        super().__init__()
        self.encoder = encoder
        self.model = models.MLP(hidden_size, 2, outputs=actions_dim)

    def forward(self, x, detach=False, deterministic=False):
        x = self.encoder(x, detach)
        logits = self.model(x)
        if deterministic:
            return torch.argmax(logits, dim=1)
        policy_dist = Categorical(logits=logits)
        action = policy_dist.sample()
        # Action probabilities for calculating the adapted soft-Q loss
        action_probs = policy_dist.probs
        log_prob = F.log_softmax(logits, dim=1)
        return action, log_prob, action_probs
    
class SAC(base_sac.SAC):
    def __init__(self, envs, args, writer, device, print_nets=True):
        super().__init__(envs, args, writer, device)
        if args.autotune:
            self.target_entropy = -args.target_entropy_scale * torch.log(1 / torch.tensor(envs.single_action_space.n))
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha = self.log_alpha.detach().exp()
            self.a_optimizer = optim.Adam([self.log_alpha], lr=args.alpha_lr)
        else:
            self.alpha = torch.tensor(args.alpha, device=device)

        if print_nets:
            utils.print_nets([
                ("actor", self.actor),
                ("critic", self.critic),
                ("critic_target", self.critic_target),
            ])
    
    def make_actor(self):
        return Actor(
            self.action_space.n, 
            self.actor_encoder,
            self.args.hidden_size
        )

    def make_critic(self):
        return Critic(
            self.action_space.n, 
            self.critic_encoder,
            self.args.hidden_size
        )

    def update_critic(self, data):
        with torch.no_grad():
            _, next_state_log_pi, next_state_action_probs = self.actor(data.next_observations)
            qf1_next_target, qf2_next_target = self.critic_target(data.next_observations)
            # we can use the action probabilities instead of MC sampling to estimate the expectation
            min_qf_next_target = next_state_action_probs * (
                torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            )
            # adapt Q-target for discrete Q-function
            min_qf_next_target = min_qf_next_target.sum(dim=1)
            next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * self.args.gamma * (min_qf_next_target)

        # use Q-values only for the taken actions
        qf1_values, qf2_values = self.critic(data.observations)
        qf1_a_values = qf1_values.gather(1, data.actions.long()).view(-1)
        qf2_a_values = qf2_values.gather(1, data.actions.long()).view(-1)
        qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
        qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
        qf_loss = qf1_loss + qf2_loss

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
        _, log_pi, action_probs = self.actor(data.observations)
        with torch.no_grad():
            qf1_values, qf2_values = self.critic(data.observations)
            min_qf_values = torch.min(qf1_values, qf2_values)
        # no need for reparameterization, the expectation can be calculated for discrete actions
        actor_loss = (action_probs * ((self.alpha * log_pi) - min_qf_values)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.log("losses", actor_loss=actor_loss.item(), alpha=self.alpha, policy_entropy=-log_pi.mean().item())

        if self.args.autotune:
            # re-use action probabilities for temperature loss
            alpha_loss = (action_probs.detach() * (-self.log_alpha.exp() * (log_pi + self.target_entropy).detach())).mean()

            self.a_optimizer.zero_grad()
            alpha_loss.backward()
            self.a_optimizer.step()
            self.alpha = self.log_alpha.detach().exp()

            self.log("losses", alpha_loss=alpha_loss.item())