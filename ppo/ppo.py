# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
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

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 0
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanrl"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = True
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "SlidingPuzzles-v0"
    env_configs: str = None
    """the id of the environment"""
    total_timesteps: int = 10000000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1024
    """the number of parallel game environments"""
    num_steps: int = 4
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.1
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    hidden_size: int = 512
    """the agent hidden size"""
    hidden_layers: int = 0
    """the number of hidden layers"""
    min_res: int = 7

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime: num_envs * num_steps)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime: batch_size // num_minibatches)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""

    checkpoint_load_path: str = None
    """the path to the checkpoint to load"""
    checkpoint_param_filter: str = ".*"
    """the filter to load checkpoint parameters"""
    checkpoint_every: int = 1e6

    freeze_param_filter: str = None
    """the filter to freeze parameters"""

    early_stop_patience: int = 100
    """the patience for early stopping"""


def make_env(env_id, idx, capture_video, run_name, env_configs):
    def thunk():
        env_configs['seed'] = env_configs['seed'] + idx
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array", **env_configs)
            env = gym.wrappers.RecordVideo(env, f"runs/{run_name}/videos")
        else:
            env = gym.make(env_id, **env_configs)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        img_obs = (
            min(env.observation_space.shape[-1], env.observation_space.shape[0]) in (3, 4)
            and "-ram" not in env.spec.id
        )
        if img_obs:
            env = gym.wrappers.ResizeObservation(env, env_configs.get("image_size", (84, 84)))
            env = sliding_puzzles.wrappers.ChannelFirstImageWrapper(env)
            env = sliding_puzzles.wrappers.NormalizedImageWrapper(env)
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    def __init__(self, envs, env_id, hidden_size, hidden_layers, min_res):
        super().__init__()
        img_obs = (
            min(envs.single_observation_space.shape[-1], envs.single_observation_space.shape[0]) in (3, 4)
            and "-ram" not in env_id
        )
        if img_obs:
            self.encoder = nn.Sequential(
                layer_init(nn.Conv2d(envs.single_observation_space.shape[0], 32, 8, stride=4)),
                nn.ReLU(),
                layer_init(nn.Conv2d(32, 64, 4, stride=2)),
                nn.ReLU(),
                layer_init(nn.Conv2d(64, 64, 3, stride=1)),
                nn.ReLU(),
                nn.Flatten(),
                layer_init(nn.Linear(64 * min_res * min_res, hidden_size)),
                nn.ReLU(),
            )
        else:
            self.encoder = nn.Sequential(
                nn.Flatten(start_dim=1),
                layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), hidden_size)),
                nn.ReLU(),
            )

        self.critic = nn.Sequential(
            *[
                layer_init(nn.Linear(hidden_size, hidden_size)),
                nn.ReLU(),
            ] * hidden_layers,
            layer_init(nn.Linear(hidden_size, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            *[
                layer_init(nn.Linear(hidden_size, hidden_size)),
                nn.ReLU(),
            ] * hidden_layers,
            layer_init(nn.Linear(hidden_size, envs.single_action_space.n), std=0.01),
        )

    def get_value(self, x):
        x = self.encoder(x)
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        x = self.encoder(x)
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)


def save_checkpoint(agent, optimizer, global_step, run_name):
    checkpoint_path = f"runs/{run_name}/checkpoint_{global_step}.pth"
    torch.save({
        'agent_state_dict': agent.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'global_step': global_step,
    }, checkpoint_path)
    print(f"Checkpoint saved at {checkpoint_path}")

def load_checkpoint(agent, optimizer, checkpoint_path, param_filter):
    print(f"\nLoading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    print("Checkpoint params: ", checkpoint['agent_state_dict'].keys())
    print("Loading params matching: ", param_filter)
    regex = re.compile(param_filter)
    filtered_state_dict = {k: v for k, v in checkpoint['agent_state_dict'].items() if regex.match(k)}
    agent_state_dict = agent.state_dict()
    agent_state_dict.update(filtered_state_dict)
    agent.load_state_dict(agent_state_dict)
    if len(filtered_state_dict) == len(checkpoint['agent_state_dict']):
        return checkpoint['global_step']
    else:
        print("Loaded params: ", filtered_state_dict.keys())
        return 0


def main():
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = int(args.total_timesteps // args.batch_size)
    if args.seed == 0:
        print("Setting seed to random")
        args.seed = random.randint(1, 1000000)
    print("Seed:", args.seed)

    args.env_configs = json.loads(args.env_configs) if args.env_configs else {}
    if not args.env_configs.get('seed'):
        print("Setting env seed to args.seed:", args.seed)
        args.env_configs['seed'] = args.seed
    if "slidingpuzzle" in args.env_id.lower() and not args.env_configs.get('image_pool_seed'):
        print("Setting image_pool_seed to seed:", args.env_configs['seed'])
        args.env_configs['image_pool_seed'] = args.env_configs['seed']

    if "slidingpuzzle" not in args.env_id.lower():
        args.exp_name += "_" + args.env_id.replace("/", "").replace("-", "").lower()
    if "w" in args.env_configs:
        args.exp_name += f"_w{args.env_configs['w']}"
    if "variation" in args.env_configs:
        args.exp_name += f"_{args.env_configs['variation']}"
    if "image_folder" in args.env_configs:
        args.exp_name += f"_{args.env_configs['image_folder'].split('/')[-1].replace('_', '').replace('-', '').lower()}"
    if "image_pool_size" in args.env_configs:
        args.exp_name += f"_p{args.env_configs['image_pool_size']}"
    run_name = f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}-{args.exp_name}_{args.seed}"

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
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    with open(f"runs/{run_name}/config.yaml", "w") as f:
        yaml.dump(vars(args), f)

    print("Configs:")
    print(json.dumps(dict(sorted(vars(args).items())), indent=2))

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # increase file descriptor limits so we can run many async envs
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
    print(f"Soft file descriptor limit: {soft}, Hard limit: {hard}")
    # env setup
    envs = gym.vector.AsyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name, args.env_configs) for i in range(args.num_envs)],
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    if "slidingpuzzle" in args.env_id.lower():
        print("Checking SlidingPuzzle envs")
        envs.reset()

        if args.env_configs.get("variation") == "image":
            env_images = envs.get_attr("images")
            for i, images in enumerate(env_images[1:]):
                assert images == env_images[i], f"All environments should have the same image list. Got: {env_images[i]} vs {images}"
        else:
            print("Variation is not image")

        env_states = envs.get_attr("state")
        assert not all(np.array_equal(env_states[i], state) for i, state in enumerate(env_states[1:])), "All environment states are identical."

    agent = Agent(envs, args.env_id, args.hidden_size, args.hidden_layers, args.min_res).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    print("Device:", device)
    print(agent)
    print(optimizer)
    # Count and print the number of parameters in the agent
    total_params = sum(p.numel() for p in agent.parameters())
    print(f"Total number of parameters in the agent: {total_params:,}")

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    success_deque = collections.deque(maxlen=args.num_envs)
    return_deque = collections.deque(maxlen=args.num_envs)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs = torch.Tensor(envs.reset(seed=args.seed)[0]).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    success_rate = 0
    early_stop_counter = 0
    next_checkpoint = args.checkpoint_every
    if args.checkpoint_load_path:
        global_step = load_checkpoint(agent, optimizer, args.checkpoint_load_path, args.checkpoint_param_filter)
    else:
        save_checkpoint(agent, optimizer, global_step, run_name)

    if args.freeze_param_filter:
        print(f"\nFreezing parameters matching: {args.freeze_param_filter}")
        for name, param in agent.named_parameters():
            if re.match(args.freeze_param_filter, name):
                param.requires_grad = False
                print(name)
        print()

    pbar = tqdm(range(1, args.num_iterations + 1), desc="Iterations")
    for iteration in pbar:
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
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                        success_deque.append(info.get("is_success", 0))
                        return_deque.append(info["episode"]["r"])

        if len(success_deque) > 0:
            success_rate = float(sum(success_deque) / args.num_envs)
            writer.add_scalar("charts/rolling_success_rate", success_rate, global_step)

            mean_return = float(sum(return_deque) / len(return_deque))
            if len(return_deque) >= args.num_envs:
                writer.add_scalar("charts/rolling_mean_return", mean_return, global_step)

            pbar.set_postfix_str(f"step={global_step}, return={mean_return:.2f}, success={success_rate:.2f}")

            if success_rate == 1 and len(success_deque) == args.num_envs:
                early_stop_counter += 1
            else:
                early_stop_counter = 0

            if args.early_stop_patience and early_stop_counter >= args.early_stop_patience:
                print("Early stopping")
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

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
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

        if global_step >= next_checkpoint:
            save_checkpoint(agent, optimizer, global_step, run_name)
            next_checkpoint = global_step + args.checkpoint_every

    envs.close()
    writer.close()

    return {"success_rate": success_rate}


if __name__ == "__main__":
    print(json.dumps(main()))
