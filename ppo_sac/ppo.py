# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
import argparse
import collections
import datetime
import json
import os
import random
import re
import resource
import time
from dataclasses import dataclass
import yaml

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torchvision import transforms

from stable_baselines3.common.atari_wrappers import (  # isort:skip
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)

import sliding_puzzles
import environments
import models
import utils
import representation

def add_args(parser):
    parser.add_argument("--learning-rate", type=float, default=2.5e-4, help="the learning rate of the optimizer")
    parser.add_argument("--num-envs", type=int, default=64, help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=16, help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=bool, action=argparse.BooleanOptionalAction, help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gamma", type=float, default=0.99, help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=4, help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=4, help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=bool, action=argparse.BooleanOptionalAction, help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.1, help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=bool, action=argparse.BooleanOptionalAction, help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.01, help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5, help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5, help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None, help="the target KL divergence threshold")

    # to be filled in runtime
    parser.add_argument("--batch-size", type=int, default=0, help="the batch size (computed in runtime: num_envs * num_steps)")
    parser.add_argument("--minibatch-size", type=int, default=0, help="the mini-batch size (computed in runtime: batch_size // num_minibatches)")
    parser.add_argument("--num-iterations", type=int, default=0, help="the number of iterations (computed in runtime)")

class Agent(nn.Module):
    def __init__(self, envs, env_id, hidden_size, hidden_layers, min_res):
        super().__init__()
        img_obs = (
            min(envs.single_observation_space.shape[-1], envs.single_observation_space.shape[0]) in (3, 4)
            and "-ram" not in env_id
        )
        if img_obs:
            base_encoder = models.ImageEncoder(
                n_channels=envs.single_observation_space.shape[0],
                min_res=min_res,
                dropout=0,
            )
        else:
            base_encoder = models.StateEncoder(
                state_size=np.array(envs.single_observation_space.shape).prod(),
                hidden_size=hidden_size,
                dropout=0
            )
        self.encoder = models.Encoder(
            base_encoder,
            hidden_size,
            variational=False
        )

        self.critic = models.MLP(hidden_size, hidden_layers, outputs=1)
        self.actor = models.MLP(hidden_size, hidden_layers, outputs=envs.single_action_space.n)
        self.apply(models.layer_init)

    def get_value(self, x):
        x = self.encoder(x)
        return self.critic(x)

    def forward(self, x, action=None, deterministic=False):
        x = self.encoder(x)
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if deterministic:
            return probs.mode()
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(conflict_handler='resolve')
    utils.add_args(parser)
    representation.add_args(parser)
    add_args(parser)
    args = parser.parse_args()
    args = utils.fix_args_pre(args)
    args = utils.fix_args_post(args)

    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = int(args.total_timesteps // args.batch_size)

    args.exp_name, run_name = utils.get_exp_name(args)
    writer = SummaryWriter(f"runs/{run_name}")

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    if torch.backends.mps.is_available() and args.cuda:
        device = torch.device("mps")
    elif torch.cuda.is_available() and args.cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # increase file descriptor limits so we can run many async envs
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
    print(f"Soft file descriptor limit: {soft}, Hard limit: {hard}")
    # env setup
    vecenv = gym.vector.AsyncVectorEnv if args.async_envs else gym.vector.SyncVectorEnv
    get_envs = lambda n: [environments.make(args.env_id, i, args.capture_video, run_name, args.env_configs, args.gamma) for i in range(n)]
    envs = vecenv(get_envs(args.num_envs))
    eval_envs = vecenv(get_envs(args.num_eval_envs)) if args.eval_every > 0 else None
    ood_envs = None
    if "slidingpuzzle" in args.env_id.lower():
        if args.ood_eval:
            env_make_list = []
            for i in range(args.num_eval_envs):
                ood_seed = random.randint(0, 1000000)
                while ood_seed >= args.env_configs["seed"] and ood_seed < args.env_configs["seed"] + args.num_envs:
                    ood_seed = random.randint(0, 1000000)
                ood_env_configs = {**args.env_configs, "seed": ood_seed, "image_pool_seed": ood_seed}
                env_make_list.append(environments.make(args.env_id, i, args.capture_video, run_name, ood_env_configs, args.gamma))
            ood_envs = vecenv(env_make_list)
        
        args.env_configs["images"] = envs.get_attr("images")[0]
    environments.check(args, envs, eval_envs, ood_envs)

    agent = Agent(envs, args.env_id, args.hidden_size, args.hidden_layers, args.min_res).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    utils.print_nets([
        ("agent", agent),
    ])

    utils.save_args(args, run_name)
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            group=args.exp_name,
            monitor_gym=True,
            save_code=True,
        )
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    print(f"Starting run {run_name}")
    global_step = 0
    start_time = time.time()
    next_obs = torch.Tensor(envs.reset(seed=args.seed)[0]).to(device) / 255.0
    next_done = torch.zeros(args.num_envs).to(device)

    next_log = 0
    next_checkpoint = 0
    next_eval = 0
    early_stop_counter = 0

    if args.checkpoint_load_path:
        global_step = utils.load_checkpoint(
            args.checkpoint_load_path,
            args.checkpoint_param_filters,
            device=device,
            agent=agent,
        )

    if args.freeze_param_filter:
        print(f"\nFreezing parameters matching: {args.freeze_param_filter}")
        for name, param in agent.named_parameters():
            if re.match(args.freeze_param_filter, name):
                param.requires_grad = False
                print(name)
        print()

    pbar = tqdm(range(1, args.num_iterations + 1), desc="Iterations")
    for iteration in pbar:
        if args.checkpoint_every > 0 and global_step >= next_checkpoint:
            next_checkpoint = global_step + args.checkpoint_every
            utils.save_checkpoint(run_name, global_step, agent=agent, optimizer=optimizer)

        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device) / 255.0, torch.Tensor(next_done).to(device)

            early_stop_counter = utils.log_stats(args, writer, infos, global_step, iteration, pbar, early_stop_counter)

        if args.early_stop_patience > 0 and early_stop_counter >= args.early_stop_patience:
            print(f"Early stopping after {global_step} steps")
            break

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent(b_obs[mb_inds], b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    utils.save_checkpoint(run_name, global_step, agent=agent, optimizer=optimizer)

    envs.close()
    writer.close()
