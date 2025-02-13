import gymnasium as gym
import numpy as np
from stable_baselines3.common.atari_wrappers import (  # isort:skip
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)

import utils


class Float32ObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super(Float32ObservationWrapper, self).__init__(env)
        self.observation_space = gym.spaces.Box(
            low=self.observation_space.low,
            high=self.observation_space.high,
            shape=self.observation_space.shape,
            dtype=np.float32
        )

    def observation(self, observation):
        return observation.astype(np.float32)


def make(env_id, idx, capture_video, run_name, env_configs, gamma=None):
    """
    Make an environment.
    Image observations are not normalized.
    """
    def thunk():
        configs = env_configs.copy()
        render_observation = configs.pop("render_observation", False)
        dmc_action_repeat = configs.pop("dmc_action_repeat", 2)
        reward_scale = configs.pop("reward_scale", 1.0)

        if "slidingpuzzle" in env_id.lower():
            import sliding_puzzles
            image_size = configs.get("image_size", 84)
            if "seed" in configs:
                configs['seed'] = configs['seed'] + idx
        elif "dm_control" in env_id:
            import dmc_utils
        else:
            image_size = configs.pop("image_size", 84)
            seed = configs.pop("seed", None)

        if "noframeskip" in env_id.lower():
            import ale_py
            gym.register_envs(ale_py)

        if capture_video and idx == 0:
            configs["render_mode"] = "rgb_array"
            env = gym.make(env_id, **configs)
            env = gym.wrappers.RecordVideo(env, f"runs/{run_name}/videos")
        else:
            env = gym.make(env_id, **configs)
        env = gym.wrappers.RecordEpisodeStatistics(env)

        if render_observation:
            env = gym.wrappers.AddRenderObservation(env)

        if "dm_control" in env_id:
            if dmc_action_repeat > 0:
                env = dmc_utils.ActionRepeatWrapper(env, dmc_action_repeat)

            env = gym.wrappers.ClipReward(env, min_reward=-1, max_reward=1)  # curl
            # env = gym.wrappers.NormalizeReward(env, gamma=gamma)
            # env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
            # env = gym.wrappers.ClipAction(env)
            env = gym.wrappers.FrameStackObservation(env, 3)

            if render_observation:
                # env = sliding_puzzles.wrappers.NormalizedImageWrapper(env)
                env = dmc_utils.FrameStackToChannelAxisWrapper(env)
            else:
                # env = gym.wrappers.NormalizeObservation(env)
                # env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
                env = gym.wrappers.FlattenObservation(env)

        elif "noframeskip" in env_id.lower():  # atari
            env = NoopResetEnv(env, noop_max=30)
            env = MaxAndSkipEnv(env, skip=4)
            env = EpisodicLifeEnv(env)
            if "FIRE" in env.unwrapped.get_action_meanings():
                env = FireResetEnv(env)
            env = ClipRewardEnv(env)
            env = gym.wrappers.ResizeObservation(env, (image_size, image_size))
            env = gym.wrappers.GrayscaleObservation(env)
            env = gym.wrappers.FrameStackObservation(env, 4)
            if seed is not None:
                env.action_space.seed(seed)

        elif utils.is_img_obs(env.observation_space):  # includes sliding puzzle
            if max(env.observation_space.shape[0], env.observation_space.shape[-1]) != image_size:
                env = gym.wrappers.ResizeObservation(env, (image_size, image_size))
            env = sliding_puzzles.wrappers.ChannelFirstImageWrapper(env)

        else:
            env = gym.wrappers.FlattenObservation(env)
            if env.observation_space.dtype != np.float32:
                env = Float32ObservationWrapper(env)

        if reward_scale != 1.0:
            env = gym.wrappers.TransformReward(env, lambda reward: reward * reward_scale)

        if idx == 0:
            print(env)
            print("Observation Space", env.observation_space)
            print("Action Space", env.action_space)
        return env

    return thunk

def check(args, train_envs, eval_envs=None, ood_envs=None):
    if "slidingpuzzle" in args.env_id.lower():
        print("Checking SlidingPuzzle envs")
        for envs in [train_envs, eval_envs]:
            if envs is None:
                continue
            envs.reset()

            if args.env_configs.get("variation") == "image":
                env_images = envs.get_attr("images")
                for i, images in enumerate(env_images[1:]):
                    assert images == env_images[i], f"All environments should have the same image list. Got: {env_images[i]} vs {images}"
            else:
                print("Variation is not image")

            if args.num_envs > 1:
                env_states = envs.get_attr("state")
                assert not all(np.array_equal(env_states[i], state) for i, state in enumerate(env_states[1:])), "All environment states are identical."

        if args.env_configs.get("variation") == "image":
            env_images = train_envs.get_attr("images")
            for i, images in enumerate(env_images[1:]):
                assert images == env_images[i], f"All environments should have the same image list. Got: {env_images[i]} vs {images}"
            if eval_envs is not None:
                eval_env_images = eval_envs.get_attr("images")
                for eval_images in eval_env_images:
                    for i in range(len(eval_images)):
                        assert eval_images[i] in env_images[0], f"In distribution eval envs should have the same image list as training envs. Got: {eval_env_images[i]} vs {env_images[i]}"

            if ood_envs is not None:
                ood_env_images = ood_envs.get_attr("images")
                for ood_images in ood_env_images:
                    for i in range(len(ood_images)):
                        assert ood_images[i] not in env_images[0], f"Out of distribution eval envs should have different image list from training envs. Got: {ood_env_images[i]} vs {env_images[i]}"
    else:
        print("Not a sliding puzzle env. Not checking.")
