import gym
import gym_platform
import math
import numpy as np
from gym.envs.registration import EnvSpec

class PlatformWrapper(object):
    def __init__(self):
        self.wrapped_env = gym.make('Platform-v0')
        self.observation_space = self.wrapped_env.observation_space[0]
        self.action_space = gym.spaces.Tuple((
            gym.spaces.Discrete(3), gym.spaces.Box(shape=(1,), low=-1.0, high=1.0)
        ))
        self.reward_range = (-math.inf, math.inf)
        self.metadata = None
        self.spec = EnvSpec(id="PlatformWrappedType1-v0")

        self.head_infos = [
            {"type": "categorical", "out_dim": 3},
            {"type": "normal", "out_dim": 1}
        ]
        self.autoregressive_maps = [
            [-1],
            [-1, 0]
        ]
        self.action_type_masks = [
            [1],
            [1],
            [1]
        ]

    def reset(self):
        return self.wrapped_env.reset()[0]

    def step(self, action):
        ac_type = action[0]
        ac_value = action[1]
        while isinstance(ac_type, list) or isinstance(ac_type, np.ndarray):
            ac_type = ac_type[0]
        ac_type = int(np.squeeze(ac_type))
        while isinstance(ac_value, list) or isinstance(ac_value, np.ndarray):
            ac_value = ac_value[0]
        ac_low = self.wrapped_env.action_space[1][ac_type].low
        ac_high = self.wrapped_env.action_space[1][ac_type].high
        transformed_ac_value = ac_low + ((ac_high - ac_low) / (2.0))*(ac_value - ac_low)
        ac_values = [np.array(0.0), np.array(0.0), np.array(0.0)]
        ac_values[ac_type] = transformed_ac_value
        final_action = (ac_type, ac_values)
        state, reward, done, info = self.wrapped_env.step(final_action)
        info["available_actions"] = self.get_available_actions()
        return state[0], reward, done, info

    def get_available_actions(self):
        avail_acs_all = []
        for i in range(len(self.head_infos)):
            ac_dim = self.head_infos[i]["out_dim"]
            avail_acs_all.append(np.ones((ac_dim,)))
        return avail_acs_all

    def seed(self, seed):
        self.wrapped_env.seed(seed)

    def render(self):
        return self.wrapped_env.render()