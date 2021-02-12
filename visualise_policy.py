import argparse
import os
import time
import torch

from core.envs import make_vec_envs
from core.utils import get_render_func, get_vec_normalize

parser = argparse.ArgumentParser(description='RL')
parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
parser.add_argument('--env-name', default='PongNoFrameskip-v4', help='environment to train on (default: PongNoFrameskip-v4)')
parser.add_argument('--load-dir', default='./trained_models/', help='directory to save agent logs (default: ./trained_models/)')
parser.add_argument('--non-det', action='store_true', default=False, help='whether to use a non-deterministic policy')
parser.add_argument('--delay', default=0.1, help='delay between frames')
args = parser.parse_args()
args.det = not args.non_det

env = make_vec_envs(args.env_name, args.seed + 1000, 1, None, None, device='cpu', allow_early_resets=False)

# Get a render function
render_func = get_render_func(env)

# We need to use the same statistics for normalization as used in training
actor_critic, obs_rms = torch.load(os.path.join(args.load_dir, args.env_name + ".pt"), map_location='cpu')

vec_norm = get_vec_normalize(env)
if vec_norm is not None:
    vec_norm.eval()
    vec_norm.obs_rms = obs_rms

recurrent_hidden_states = torch.zeros(1, actor_critic.recurrent_hidden_state_size)
masks = torch.zeros(1, 1)

obs = env.reset()
render_func()

while True:
    with torch.no_grad():
        value, action, _, recurrent_hidden_states = actor_critic.act(
            obs, recurrent_hidden_states, masks, deterministic=args.det)

    # Obser reward and next obs
    obs, reward, done, _ = env.step(action)

    masks.fill_(0.0 if done else 1.0)

    render_func()
    time.sleep(args.delay)