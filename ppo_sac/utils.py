import argparse
import collections
from dataclasses import dataclass
import datetime
import json
import random
import re
import os
import numpy as np
from stable_baselines3.common.buffers import ReplayBuffer as RB
import torch
from tqdm import tqdm
import sys
import yaml

import augmentations


def add_args(parser):
    parser.add_argument("-f", "--fail-fast", action="store_true", help="fail fast if any error occurs")
    parser.add_argument("--exp-name", type=str, default=os.path.basename(os.path.abspath(sys.argv[0]))[: -len(".py")], help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=0, help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=bool, default=True, action=argparse.BooleanOptionalAction, help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=bool, default=True, action=argparse.BooleanOptionalAction, help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=bool, default=True, action=argparse.BooleanOptionalAction, help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--is-sweep", type=bool, default=False, action=argparse.BooleanOptionalAction, help="if toggled, this experiment will be treated as a sweep")
    parser.add_argument("--wandb-project-name", type=str, default="cleanrl-sac", help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default="bryanoliveira", help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=bool, default=True, action=argparse.BooleanOptionalAction, help="whether to capture videos of the agent performances (check out `videos` folder)")
    parser.add_argument("--env-id", type=str, default="SlidingPuzzles-v0", help="the id of the environment")
    parser.add_argument("--env-configs", type=str, default=None)
    parser.add_argument("--reward-scale", type=float, default=1.0)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--async-envs", type=bool, default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--total-timesteps", type=int, default=int(1e7), help="total timesteps of the experiments")
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "rmsprop"])
    parser.add_argument("--min-res", type=int, default=7)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--hidden-layers", type=int, default=0)
    parser.add_argument("--max-grad-norm", type=float, default=10)
    parser.add_argument("--checkpoint-load-path", type=str, default=None, help="the path to the checkpoint to load")
    parser.add_argument("--checkpoint-param-filters", type=str, default=None, help="the filter to load checkpoint parameters")
    parser.add_argument("--use-checkpoint-images", type=bool, default=False, action=argparse.BooleanOptionalAction, help="whether to use the same images used to train the checkpoint")
    parser.add_argument("--num-checkpoints", type=int, default=10, help="the number of checkpoints to save, set to 0 to disable")
    parser.add_argument("--checkpoint-every", type=int, default=0, help="the number of steps between checkpoints, set to 0 to disable")
    parser.add_argument("--freeze-param-filter", type=str, default=None, help="the filter to freeze parameters")
    parser.add_argument("--early-stop-patience", type=int, default=100, help="the patience for early stopping, in episodes")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=0, help="the frequency to evaluate the agent, in steps")
    parser.add_argument("--num-eval-envs", type=int, default=10)
    parser.add_argument("--ood-eval", type=bool, default=False, action=argparse.BooleanOptionalAction)

def get_exp_name(args):
    exp_name = args.exp_name
    if args.use_reward_model:
        exp_name += "_rew"
    if args.use_transition_model:
        exp_name += "_tr"
        if args.probabilistic_transition_model:
            exp_name += "p"
    if args.use_curl_loss:
        exp_name += "_curl"
        if args.contrastive_positives == "temporal":
            exp_name += "t"
    if args.use_reconstruction_loss:
        exp_name += "_rec"
        if args.variational_reconstruction:
            exp_name += "v"
    if args.use_dbc_loss:
        exp_name += "_dbc"
    if args.use_spr_loss:
        exp_name += "_spr"
    if args.use_data_augmentation_loss:
        exp_name += "_da"
    elif args.apply_data_augmentation:
        exp_name += "a"

    if "slidingpuzzle" not in args.env_id.lower():
        exp_name += "_" + args.env_id.replace("/", "_").replace("-", "_").lower()
    if "w" in args.env_configs:
        exp_name += f"_w{args.env_configs['w']}"
    if "variation" in args.env_configs:
        exp_name += f"_{args.env_configs['variation']}"
    if "image_folder" in args.env_configs:
        exp_name += f"_{args.env_configs['image_folder'].split('/')[-1].replace('_', '').replace('-', '').lower()}"
    if "image_pool_size" in args.env_configs:
        exp_name += f"_p{args.env_configs['image_pool_size']}"

    return exp_name, f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}-{exp_name}_{args.seed}"


def fix_args_pre(args):
    if args.seed == 0:
        print("Setting seed to random")
        args.seed = random.randint(1, 1000000)
    print("Seed:", args.seed)

    if args.is_sweep:
        args.track = False
    
    if args.fail_fast:
        print("--- FAILING FAST ---")
        args.is_sweep = False
        args.track = False
        args.num_envs = 3
        args.learning_starts = 500
        args.batch_size = 4
        args.buffer_size = 100
        args.log_every = 1
        args.eval_every = 1
        args.num_eval_envs = 2

    if args.num_checkpoints > 0:
        args.checkpoint_every = args.total_timesteps // args.num_checkpoints
        print(f"Setting checkpoint every {args.checkpoint_every} steps")
    args.checkpoint_param_filters = json.loads(args.checkpoint_param_filters) if args.checkpoint_param_filters else {}

    args.env_configs = json.loads(args.env_configs) if args.env_configs else {}
    if not args.env_configs.get('seed'):
        print("Setting env seed to args.seed:", args.seed)
        args.env_configs['seed'] = args.seed

    if not args.env_configs.get('reward_scale'):
        args.env_configs['reward_scale'] = args.reward_scale

    if "slidingpuzzle" in args.env_id.lower():
        if not args.env_configs.get('image_pool_seed'):
            print("Setting image_pool_seed to seed:", args.env_configs['seed'])
            args.env_configs['image_pool_seed'] = args.env_configs['seed']
        if not args.env_configs.get('w'):
            print("Setting w to 3")
            args.env_configs['w'] = 3
        if not args.env_configs.get('variation'):
            print("Setting variation to image")
            args.env_configs['variation'] = "image"
        if args.env_configs['variation'] == "image" and not args.env_configs.get('image_folder'):
            print("Setting image_folder to imagenet-1k")
            args.env_configs['image_folder'] = "imagenet-1k"
        if args.env_configs['variation'] == "image" and not args.env_configs.get('image_pool_size'):
            print("Setting image_pool_size to 1")
            args.env_configs['image_pool_size'] = 1
        if args.use_checkpoint_images:
            checkpoint_exp_folder = os.path.dirname(args.checkpoint_load_path)
            print("Using same images from checkpoint:", checkpoint_exp_folder)
            configs = yaml.load(open(checkpoint_exp_folder + "/config.yaml", "r"), Loader=yaml.FullLoader)
            args.env_configs['image_folder'] = configs['env_configs']['image_folder']
            args.env_configs['image_pool_size'] = configs['env_configs']['image_pool_size']
            args.env_configs['image_pool_seed'] = configs['env_configs']['image_pool_seed']
            args.env_configs['images'] = configs['env_configs']['images']

    if args.use_reward_loss:
        if not args.use_reward_model:
            print("[REW] Overwriting use_reward_model to True")
            args.use_reward_model = True
    if args.use_transition_loss:
        if not args.use_transition_model:
            print("[TR] Overwriting use_transition_model to True")
            args.use_transition_model = True
    if args.use_reconstruction_loss:
        if not args.train_decoder:
            print("[REC] Overwriting train_decoder to True")
            args.train_decoder = True
    if (
        (args.use_curl_loss and args.contrastive_positives == "augmented")
        or args.use_data_augmentation_loss
    ):
        if not args.apply_data_augmentation:
            print("[DA] Overwriting apply_data_augmentation to True")
            args.apply_data_augmentation = True
    if args.use_dbc_loss:
        if not args.probabilistic_transition_model:
            print("[DBC] Overwriting probabilistic_transition_model to True")
            args.probabilistic_transition_model = True
        if not args.use_transition_model:
            print("[DBC] Overwriting use_transition_model to True")
            args.use_transition_model = True
        if not args.use_transition_loss:
            print("[DBC] Overwriting use_transition_loss to True")
            args.use_transition_loss = True
        if not args.use_reward_model:
            print("[DBC] Overwriting use_reward_model to True")
            args.use_reward_model = True
        if not args.use_reward_loss:
            print("[DBC] Overwriting use_reward_loss to True")
            args.use_reward_loss = True
    if args.use_spr_loss:
        assert args.apply_data_augmentation or args.encoder_dropout > 0, "[SPR] Encoder dropout must be > 0 or should use augmentation"
        if not args.use_transition_model:
            print("[SPR] Overwriting use_transition_model to True")
            args.use_transition_model = True

    if not args.env_configs.get('image_size'):
        args.env_configs['image_size'] = 100 if args.apply_data_augmentation and "crop" in args.augmentations else 84
        print(f"Setting image_size to {args.env_configs['image_size']}")

    return args

def fix_args_post(args):
    """Fixes args available after algorithm selection"""
    if not args.independent_encoder:
        if hasattr(args, "tau") and args.encoder_tau != args.tau:
            print("[INDEP] Overwriting encoder_tau to tau")
            args.encoder_tau = args.tau
        if hasattr(args, "target_frequency") and args.target_encoder_frequency != args.target_frequency:
            print("[INDEP] Overwriting target_encoder_frequency to target_frequency")
            args.target_encoder_frequency = args.target_frequency

    return args

def save_args(args, run_name):
    os.makedirs(f"runs/{run_name}", exist_ok=True)
    with open(f"runs/{run_name}/config.yaml", "w") as f:
        yaml.dump(vars(args), f)

    print("Configs:")
    print(json.dumps(dict(sorted(vars(args).items())), indent=2))

def is_img_obs(observation_space):
    """Check if observation space represents an image.
    
    Args:
        observation_space: Gym observation space
        
    Returns:
        bool: True if observation space represents an image (3D with channels), False otherwise
        
    Notes:
        Handles both regular image observations (C,H,W) and frame-stacked observations (N*C,H,W)
        where N is number of stacked frames and C is base number of channels
    """
    # Check if space has 3 dimensions (channels, height, width) 
    if len(observation_space.shape) != 3:
        return False
        
    # Get number of channels
    channels = observation_space.shape[0]
    
    # Check if channels is divisible by standard image channel counts
    # to handle frame stacking (e.g. 4 frames * 3 RGB channels = 12 channels)
    for base_channels in [1, 3, 4]:  # Standard channel counts: mono, RGB, RGBA
        if channels % base_channels == 0:
            return True
            
    return False


success_deque = None
return_deque = None
def log_stats(args, writer, infos, global_step, num_updates, pbar=None, early_stop_counter=0):
    global success_deque, return_deque
    if success_deque is None:
        success_deque = collections.deque(maxlen=args.num_envs)
        return_deque = collections.deque(maxlen=args.num_envs)

    postfix_str = f"update={num_updates}"
    should_update = False
    if "episode" in infos:
        for i, done in enumerate(infos["_episode"]):
            if done:
                should_update = True
                writer.add_scalar("charts/episodic_return", infos["episode"]["r"][i], global_step)
                writer.add_scalar("charts/episodic_length", infos["episode"]["l"][i], global_step)
                return_deque.append(infos["episode"]["r"][i])
                if "is_success" in infos:
                    success_deque.append(infos["is_success"][i])
                
    if should_update:
        if len(success_deque) >= args.num_envs:
            success_rate = float(sum(success_deque) / len(success_deque))
            postfix_str += f", success={success_rate:.2f}"
            writer.add_scalar("charts/rolling_success_rate", success_rate, global_step)

            if args.early_stop_patience > 0:
                if success_rate == 1 and len(success_deque) == args.num_envs:
                    early_stop_counter += 1
                else:
                    early_stop_counter = 0

        if len(return_deque) >= args.num_envs:
            mean_return = float(sum(return_deque) / len(return_deque))
            postfix_str += f", return={mean_return:.2f}"
            writer.add_scalar("charts/rolling_mean_return", mean_return, global_step)

        if pbar is not None:
            pbar.set_postfix_str(postfix_str)

    return early_stop_counter


def evaluate(agent, envs, writer=None, global_step=0, suffix="", deterministic=False):
    # agent.eval()
    device = next(agent.parameters()).device
    obs, _ = envs.reset()
    episodes = 0
    returns = 0.0
    lengths = 0.0
    successes = 0.0
    pbar = tqdm(range(envs.get_attr("max_steps")[0]), desc="Eval steps")
    while episodes < envs.num_envs:
        obs = torch.Tensor(obs).to(device) / 255.0
        if obs.shape[-1] != 84:
            obs = torch.nn.functional.interpolate(obs, size=(84, 84), mode='bilinear', align_corners=False)
        pbar.update(1)

        with torch.no_grad():
            agent_actions = agent(obs, deterministic=deterministic)
            if not deterministic:
                agent_actions = agent_actions[0]
            agent_actions = agent_actions.detach().cpu().numpy()

        obs, reward, terminated, truncated, info = envs.step(agent_actions)

        if "episode" in info:
            for i, done in enumerate(info["_episode"]):
                if done:
                    returns += info["episode"]["r"][i]
                    lengths += info["episode"]["l"][i]
                    episodes += 1
                    pos_str = f"r={(returns/episodes):.2f}"
                    if "is_success" in info:
                        pos_str += f", s={(successes/episodes):.2f}"
                        successes += info["is_success"][i]
                    pbar.set_postfix_str(pos_str)

    # agent.train()

    if writer is not None:
        writer.add_scalar(f"eval{suffix}/mean_return", returns / episodes, global_step)
        writer.add_scalar(f"eval{suffix}/mean_length", lengths / episodes, global_step)
        writer.add_scalar(f"eval{suffix}/mean_successes", successes / episodes, global_step)
    return returns, lengths, successes, episodes


def print_nets(nets):
    for name, net in nets:
        print(f"--- {name} ---")
        print(net)
        print("Trainable parameters:", sum(p.numel() for p in net.parameters() if p.requires_grad))
        print("Device:", next(net.parameters()).device)
        print("---------------")


def save_checkpoint(run_name, global_step, **objects):
    checkpoint_path = f"runs/{run_name}/checkpoint_{global_step}.pth"
    save_dict = {'global_step': global_step}
    for key, obj in objects.items():
        save_dict[key] = obj.state_dict()
    torch.save(save_dict, checkpoint_path)
    print(f"Checkpoint saved at {checkpoint_path}")


def load_checkpoint(checkpoint_path, param_filters=None, device=None, **objects):
    print(f"\nLoading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    filtered_any = False
    if not param_filters:
        param_filters = {}
    for key, module in objects.items():
        if key not in checkpoint:
            print(f"Checkpoint does not contain {key}")
            continue
        module_state_dict = checkpoint[key]
        if param_filters.get(key):
            filtered_any = True
            print("Checkpoint params: ", checkpoint[key].keys())
            print("Unloading params not matching: ", param_filters[key])
            regex = re.compile(param_filters[key])
            filtered_state_dict = {k: v for k, v in module.state_dict().items() if not regex.match(k)}
            print("Unloaded params: ", filtered_state_dict.keys())
            module_state_dict.update(filtered_state_dict)
        module.load_state_dict(module_state_dict)
        
    if filtered_any:
        return 0
    else:
        return checkpoint['global_step']


@dataclass
class ReplayBufferSamples:
    observations: torch.Tensor
    actions: torch.Tensor
    next_observations: torch.Tensor
    dones: torch.Tensor
    rewards: torch.Tensor
    posterior: torch.Tensor = None
    multistep_observations: torch.Tensor = None
    multistep_actions: torch.Tensor = None
    multistep_next_observations: torch.Tensor = None
    multistep_dones: torch.Tensor = None
    multistep_rewards: torch.Tensor = None


class ReplayBuffer(RB):
    def __init__(
        self,
        augs_str: str = "",
        return_posterior: bool = False,
        normalize_imgs: bool = True,
        horizon: int = 1,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.return_posterior = return_posterior
        self.normalize_imgs = normalize_imgs and is_img_obs(kwargs["observation_space"])
        self.horizon = horizon
        self.augmentations = [
            getattr(augmentations, aug)
            for aug in augs_str.split(",")
            if aug != ""
        ]
        print("Augmentations:", self.augmentations)

    def sample(self, *args, **kwargs) -> ReplayBufferSamples:
        if self.horizon > 1:
            return self.sample_multistep(*args, **kwargs)
        else:
            return self.sample_single(*args, **kwargs)

    def sample_single(self, *args, **kwargs) -> ReplayBufferSamples:
        data = super().sample(*args, **kwargs)
        data = {
            "observations": data.observations,
            "actions": data.actions,
            "next_observations": data.next_observations,
            "dones": data.dones,
            "rewards": data.rewards,
        }
        if self.return_posterior:
            data["posterior"] = data["observations"].clone()

        if self.normalize_imgs:
            for key in ["observations", "next_observations", "posterior"]:
                if key in data:
                    data[key] = data[key].float() / 255.0

        for aug in self.augmentations:
            if self.return_posterior:
                data["posterior"] = aug(data["posterior"])
            data["observations"] = aug(data["observations"])
            data["next_observations"] = aug(data["next_observations"])

        return ReplayBufferSamples(**data)

    def sample_multistep(self, batch_size: int) -> ReplayBufferSamples:
        # Get valid indices that don't cross episode boundaries
        last_idx = self.buffer_size if self.full else self.pos
        last_idx -= self.horizon

        # Get indices of done episodes
        done_idxs = np.where(self.dones[:last_idx, 0] == 1.0)[0]
        done_idxs = np.concatenate([done_idxs, [last_idx]])
        
        n_done = len(done_idxs)
        done_idxs_raw = done_idxs - np.arange(1, n_done+1) * self.horizon

        # Sample random starting points
        samples_raw = np.random.choice(
            last_idx - (self.horizon + 1) * n_done, 
            size=batch_size,
            replace=True
        )
        samples_raw = np.sort(samples_raw)
        
        # Find valid episode segments
        js = np.searchsorted(done_idxs_raw, samples_raw)
        offsets = done_idxs_raw[js] - samples_raw + self.horizon
        start_idxs = done_idxs[js] - offsets

        # Collect sequences
        obses, actions = [], []
        
        # we only need multistep for observations and actions
        for t in range(self.horizon):
            idx = start_idxs + t
            obses.append(self.observations[idx, 0])
            actions.append(self.actions[idx, 0])

        # Stack and convert to tensors
        obses = torch.tensor(np.stack(obses), device=self.device)
        actions = torch.tensor(np.stack(actions), device=self.device)
        next_obs = torch.tensor(self.next_observations[start_idxs, 0], device=self.device)
        rewards = torch.tensor(self.rewards[start_idxs, 0], device=self.device)
        dones = torch.tensor(self.dones[start_idxs, 0], device=self.device)

        if self.normalize_imgs:
            obses = obses.float() / 255.0
            next_obs = next_obs.float() / 255.0

        # Apply augmentations if needed
        for aug in self.augmentations:
            obses = aug(obses.reshape(self.horizon * batch_size, *obses.shape[2:]))
            obses = obses.reshape(self.horizon, batch_size, *obses.shape[1:])
            next_obs = aug(next_obs)

        data = {
            "observations": obses[0],
            "actions": actions[0],
            "next_observations": next_obs,
            "dones": dones,
            "rewards": rewards,
            "multistep_observations": obses,
            "multistep_actions": actions,
        }

        return ReplayBufferSamples(**data)
