# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_continuous_actionpy
import argparse
import random
import resource
import time

import gymnasium as gym
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter
import wandb

import algorithms
import environments
import utils
import algorithms
import representation


if __name__ == "__main__":
    parser = argparse.ArgumentParser(conflict_handler='resolve')
    utils.add_args(parser)
    representation.add_args(parser)
    args, remaining_args = parser.parse_known_args()
    args = utils.fix_args_pre(args)

    dummy_env = environments.make(args.env_id, 0, False, "dummy", args.env_configs or {})()
    if isinstance(dummy_env.action_space, gym.spaces.Discrete):
        agent_cls = algorithms.SACDiscrete
        algorithms.sac_discrete.add_args(parser)
    else:
        agent_cls = algorithms.SACContinuous
        algorithms.sac_continuous.add_args(parser)
    dummy_env.close()

    args = parser.parse_args(remaining_args, args)
    args = utils.fix_args_post(args)

    if args.is_sweep:
        wandb.init(project="cleanrl-sac-sweep", sync_tensorboard=True)
        for k, v in wandb.config.items():
            k = k.replace("-", "_")
            assert hasattr(args, k), f"Argument {k} not found"
            setattr(args, k, v)
            print(f"Overwriting {k}: {v}")

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
    eval_envs = vecenv(get_envs(args.num_eval_envs))
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

    agent = agent_cls(envs, args, writer, device)
    repr = representation.Representation(
        args,
        envs.single_action_space,
        agent.critic.encoder,
        agent.critic_target.encoder,
        writer,
        device
    )

    rb = utils.ReplayBuffer(
        augs_str=args.augmentations if args.apply_data_augmentation else "",
        return_posterior=args.use_curl_loss,
        horizon=args.spr_horizon if args.use_spr_loss else 1,
        buffer_size=args.buffer_size,
        observation_space=envs.single_observation_space,
        action_space=envs.single_action_space,
        device=device,
        n_envs=1,  # we add transitions one by one to handle autoreset
        handle_timeout_termination=False,
    )

    utils.save_args(args, run_name)
    if args.is_sweep:
        wandb.config.update(vars(args), allow_val_change=True)
    elif args.track:
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            # monitor_gym=True,
            save_code=True,
        )
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    print(f"Starting run {run_name}")
    next_log = 0
    next_checkpoint = 0
    next_eval = 0
    early_stop_counter = 0
    autoreset = np.zeros(envs.num_envs)
    global_step = 0

    if args.checkpoint_load_path:
        global_step = utils.load_checkpoint(
            args.checkpoint_load_path,
            args.checkpoint_param_filters,
            device=device,
            agent=agent,
            repr=repr,
        )
        global_step -= args.buffer_size  # will need to refill buffer
        args.learning_starts = global_step + args.buffer_size
        agent.global_step = global_step
        repr.global_step = global_step

    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    pbar = tqdm(range(args.total_timesteps), desc="Timesteps", initial=global_step)
    while global_step < args.total_timesteps:
        if args.checkpoint_every > 0 and global_step >= next_checkpoint:
            next_checkpoint = global_step + args.checkpoint_every
            utils.save_checkpoint(run_name, global_step, agent=agent, repr=repr)

        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts and not args.checkpoint_load_path:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            inputs = torch.Tensor(obs).to(device) / 255.0
            if inputs.shape[-1] != 84:
                inputs = torch.nn.functional.interpolate(inputs, size=(84, 84), mode='bilinear', align_corners=False)
            actions, _, _ = agent.actor(inputs)
            actions = actions.detach().cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        global_step += envs.num_envs
        pbar.update(envs.num_envs)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        early_stop_counter = utils.log_stats(args, writer, infos, global_step, agent.num_updates, pbar, early_stop_counter)
        if early_stop_counter >= args.early_stop_patience:
            print(f"Early stopping after {global_step} steps")
            break
        
        # TRY NOT TO MODIFY: save data to replay buffer; handle `final_observation`
        for j in range(envs.num_envs):
            if not autoreset[j]:
                rb.add(obs[j], next_obs[j], actions[j], rewards[j], terminations[j], {})

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs
        autoreset = np.logical_or(terminations, truncations)

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            data = rb.sample(args.batch_size)
            agent.update(data, global_step)
            repr.update(data, global_step)

        if args.log_every > 0 and global_step >= next_log:
            next_log = global_step + args.log_every * args.num_envs
            writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

        if args.eval_every > 0 and global_step >= next_eval:
            next_eval = global_step + args.eval_every
            utils.evaluate(agent.actor, eval_envs, writer, global_step)
            if ood_envs is not None:
                utils.evaluate(agent.actor, ood_envs, writer, global_step, suffix="_ood")

    utils.save_checkpoint(run_name, global_step, agent=agent, repr=repr)

    envs.close()
    writer.close()