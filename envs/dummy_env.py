import gym
import numpy as np
import math
from gym.envs.registration import EnvSpec

class DummyEnv(object):
    def __init__(self):
        self.observation_space = gym.spaces.Box(-1, 1, shape=(10,))
        self.action_space = gym.spaces.Discrete(4)
        self.reward_range = (-math.inf, math.inf)
        self.metadata = None
        self.spec = EnvSpec(id="DummyEnv-v0")

    def _get_state(self):
        return np.clip(np.random.randn(self.observation_space.shape[0]), -1, 1)

    def get_available_actions(self):
        avail_acs = (np.random.rand(self.action_space.n) > 0.5).astype(float)
        ind = int(np.random.randint(0, self.action_space.n))
        avail_acs[ind] = 1.0
        return avail_acs

    def reset(self):
        return self._get_state()

    def step(self, action):
        if isinstance(action, int):
            assert action < self.action_space.n
        else:
            assert action[0] < self.action_space.n

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