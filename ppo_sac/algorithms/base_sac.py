import argparse
import numpy as np
from torch import optim

import models
import utils


def add_args(parser):
    parser.add_argument("--buffer-size", type=int, default=300000, help="the replay memory buffer size")
    parser.add_argument("--batch-size", type=int, default=4096, help="the batch size of sample from the replay memory")
    parser.add_argument("--learning-starts", type=int, default=int(2e4), help="number of warmup steps to fill the replay buffer")
    parser.add_argument("--q-lr", type=float, default=3e-4, help="the learning rate of the Q network network optimizer")
    parser.add_argument("--target-frequency", type=int, default=8000, help="the frequency of updates for the target networks")
    parser.add_argument("--policy-lr", type=float, default=3e-4, help="the learning rate of the policy network optimizer")
    parser.add_argument("--policy-frequency", type=int, default=1, help="the frequency of training updates")
    parser.add_argument("--gamma", type=float, default=0.99, help="the discount factor gamma")
    parser.add_argument("--autotune", type=bool, default=True, action=argparse.BooleanOptionalAction, help="automatic tuning of the entropy coefficient")
    parser.add_argument("--alpha", type=float, default=0.2, help="Entropy regularization coefficient.")
    parser.add_argument("--alpha-lr", type=float, default=3e-4, help="the learning rate of the alpha optimizer")
    parser.add_argument("--alpha-beta", type=float, default=0.9, help="the beta of the entropy coefficient optimizer")  # 0.5


class SAC:
    def __init__(self, envs, args, writer, device):
        self.args = args
        self.writer = writer
        self.observation_space = envs.single_observation_space
        self.action_space = envs.single_action_space
        img_obs = utils.is_img_obs(self.observation_space)
        if img_obs:
            base_encoder = models.ImageEncoder(
                n_channels=self.observation_space.shape[0],
                min_res=args.min_res,
                dropout=args.encoder_dropout,
            ).to(device)
        else:
            envs.single_observation_space.dtype = np.float32
            base_encoder = models.StateEncoder(
                state_size=np.array(self.observation_space.shape).prod(),
                hidden_size=args.hidden_size,
                dropout=args.encoder_dropout
            ).to(device)

        # share cnn but keep separate projection layers
        self.actor_encoder = models.Encoder(
            base_encoder,
            args.hidden_size,
            variational=args.variational_reconstruction
        ).to(device)
        self.critic_encoder = models.Encoder(
            base_encoder,
            args.hidden_size,
            variational=args.variational_reconstruction
        ).to(device)

        self.actor = self.make_actor().to(device)
        self.critic = self.make_critic().to(device)

        self.critic_target = models.get_target(self.critic)
        self.q_optimizer = optim.Adam(list(self.critic.parameters()), lr=args.q_lr)
        self.actor_optimizer = optim.Adam(list(self.actor.parameters()), lr=args.policy_lr)

        self.num_updates = 0
        self.global_step = 0
        self.next_log = args.log_every

    def make_actor(self):
        raise NotImplementedError
    
    def make_critic(self):
        raise NotImplementedError

    def update(self, data, global_step):
        self.global_step = global_step
        self.num_updates += 1

        self.update_critic(data)

        if self.num_updates % self.args.policy_frequency == 0:  # TD 3 Delayed update support
            for _ in range(
                self.args.policy_frequency
            ):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                self.update_actor_and_alpha(data)
        
        # update the target networks
        if self.num_updates % self.args.target_frequency == 0:
            models.update_target(self.critic.Q1, self.critic_target.Q1, self.args.tau)
            models.update_target(self.critic.Q2, self.critic_target.Q2, self.args.tau)

        if self.num_updates % self.args.target_encoder_frequency == 0:
            models.update_target(self.critic.encoder, self.critic_target.encoder, self.args.encoder_tau)

        if self.num_updates >= self.next_log:
            self.log("charts", num_updates=self.num_updates)
            self.next_log = self.num_updates + self.args.log_every

    def log(self, group, **kwargs):
        if self.num_updates >= self.next_log:
            for name, value in kwargs.items():
                self.writer.add_scalar(f"{group}/{name}", value, self.global_step)

    def state_dict(self):
        return {
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.q_optimizer.state_dict(),
        }

    def load_state_dict(self, state_dict):
        self.actor.load_state_dict(state_dict['actor'])
        self.critic.load_state_dict(state_dict['critic'])
        self.critic_target.load_state_dict(state_dict['critic_target'])
        self.actor_optimizer.load_state_dict(state_dict['actor_optimizer'])
        self.q_optimizer.load_state_dict(state_dict['critic_optimizer'])