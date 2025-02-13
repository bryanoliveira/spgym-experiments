import functools
import numpy as np
import embodied


class SlidingPuzzles(embodied.Env):

    def __init__(self, variation, **kwargs):
        import sliding_puzzles

        self._env = sliding_puzzles.make(
            variation=variation, render_mode="rgb_array", **kwargs
        )
        self._variation = variation
        self._done = True

    @functools.cached_property
    def act_space(self):
        spaces = {"action": self._convert(self._env.get_wrapper_attr("action_space"))}
        spaces["reset"] = embodied.Space(bool)
        return spaces

    @functools.cached_property
    def obs_space(self):
        spaces = {
            "reward": embodied.Space(np.float32),
            "is_first": embodied.Space(bool),
            "is_last": embodied.Space(bool),
            "is_terminal": embodied.Space(bool),
            "log_success": embodied.Space(np.float32, expand=False),
        }
        if self._variation == "image":
            spaces["image"] = self._convert(
                self._env.get_wrapper_attr("observation_space")
            )
        else:
            spaces["observation"] = self._convert(
                self._env.get_wrapper_attr("observation_space")
            )
            # the render size is W x H, so the returned matrix must be H x W x 3
            spaces["image"] = embodied.Space(
                np.uint8, self._env.get_wrapper_attr("render_size")[::-1] + (3,)
            )
        return spaces

    def _obs(
        self,
        obs,
        reward,
        is_first=False,
        is_last=False,
        is_terminal=False,
        is_success=False,
    ):
        ret = {
            "reward": np.float32(reward),
            "is_first": is_first,
            "is_last": is_last,
            "is_terminal": is_terminal,
            "log_success": np.float32(is_success),
        }
        if self._variation == "image":
            ret["image"] = np.asarray(obs)
        else:
            ret["observation"] = np.asarray(obs)
            ret["image"] = self._env.render()
        return ret

    def _convert(self, space):
        if hasattr(space, "n"):
            return embodied.Space(np.int32, (), 0, space.n)
        return embodied.Space(space.dtype, space.shape, space.low, space.high)


    def step(self, action):
        if action["reset"] or self._done:
            self._done = False
            obs, _ = self._env.reset()
            return self._obs(obs, 0.0, is_first=True)

        action = action["action"]
        obs, reward, self._done, _, info = self._env.step(action)
        return self._obs(
            obs,
            reward,
            is_last=bool(self._done),
            is_terminal=bool(info.get("is_success", self._done)),
            is_success=info.get("is_success", False),
        )

    def reset(self):
        obs, info = self._env.reset()
        return self._obs(
            obs, 0.0, is_first=True, is_success=info.get("is_success", False)
        )

    def close(self):
        try:
            self._env.close()
        except Exception:
            pass