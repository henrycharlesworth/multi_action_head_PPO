import gym
import numpy as np
import math
from gym.envs.registration import EnvSpec

class DummyMultiHeadEnv(object):
    def __init__(self):
        self.observation_space = gym.spaces.Box(-1, 1, shape=(10,))
        self.action_space = gym.spaces.Tuple((
            gym.spaces.Discrete(3), gym.spaces.Discrete(4), gym.spaces.Discrete(5),
            gym.spaces.Box(shape=(2,), low=-1.0, high=1.0)
        ))
        self.reward_range = (-math.inf, math.inf)
        self.metadata = None
        self.spec = EnvSpec(id="DummyEnv-v0")

        self.head_infos = [
            {"type": "categorical", "out_dim": 3},
            {"type": "categorical", "out_dim": 4},
            {"type": "categorical", "out_dim": 5},
            {"type": "normal", "out_dim": 2}
        ]
        self.autoregressive_maps = [
            [-1],
            [-1, 0],
            [-1, 0, 1],
            [-1, 0]
        ]
        self.action_type_masks = [
            [1, 1, 0],
            [1, 1, 1],
            [0, 0, 1]
        ]

    def _get_state(self):
        return np.clip(np.random.randn(self.observation_space.shape[0]), -1, 1)

    def get_available_actions(self):
        avail_acs_all = []
        for i in range(len(self.head_infos)):
            ac_dim = self.head_infos[i]["out_dim"]
            avail_acs = np.random.rand(ac_dim) > 0.5
            ind = int(np.random.randint(0, ac_dim))
            avail_acs[ind] = 1.0
            avail_acs_all.append(avail_acs.astype(float))
        return avail_acs_all

    def reset(self):
        return self._get_state()

    def step(self, action):
        for i, ac in enumerate(action):
            ac_dim = self.head_infos[i]["out_dim"]
            if isinstance(ac, int):
                if self.head_infos[i]["type"] == "categorical":
                    assert ac < ac_dim
            else:
                if self.head_infos[i]["type"] == "categorical":
                    assert ac[0] < ac_dim

        state = self._get_state()
        reward = np.random.rand()
        if np.random.rand() < 0.1:
            done = True
        else:
            done = False
        info = {
            "available_actions": self.get_available_actions()
        }
        return state, reward, done, info

    def seed(self, seed):
        np.random.seed(seed)