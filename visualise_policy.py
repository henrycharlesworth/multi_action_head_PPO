import argparse
import os
import time
import torch

from core.envs import make_vec_envs
from core.utils import get_render_func, get_vec_normalize

parser = argparse.ArgumentParser(description='RL')
parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
parser.add_argument('--env-name', default='PongNoFrameskip-v4', help='environment to train on (default: PongNoFrameskip-v4)')
parser.add_argument('--load-dir', default='./saved_models/ppo', help='directory to save agent logs (default: ./trained_models/)')
parser.add_argument('--non-det', action='store_true', default=False, help='whether to use a non-deterministic policy')
parser.add_argument('--delay', default=0.1, help='delay between frames')
args = parser.parse_args()
args.det = not args.non_det

# We need to use the same statistics for normalization as used in training
actor_critic, obs_rms = torch.load(os.path.join(args.load_dir, args.env_name + ".pt"), map_location='cpu')
no_obs_norm = False
if obs_rms is None:
    no_obs_norm = True

env = make_vec_envs(args.env_name, args.seed + 1000, 1, None, None, device='cpu', allow_early_resets=False,
                    no_obs_norm=no_obs_norm)

# Get a render function
render_func = get_render_func(env)

if no_obs_norm == False:
    vec_norm = get_vec_normalize(env)
    if vec_norm is not None:
        vec_norm.eval()
        vec_norm.obs_rms = obs_rms

if actor_critic.is_recurrent and actor_critic.base.recurrent_type == "LSTM":
    recurrent_hidden_states = (torch.zeros(1, actor_critic.recurrent_hidden_state_size),
                               torch.zeros(1, actor_critic.recurrent_hidden_state_size))
else:
    recurrent_hidden_states = torch.zeros(1, actor_critic.recurrent_hidden_state_size)
masks = torch.zeros(1, 1)

try:
    head_infos = env.get_attr("head_infos")[0]
    multi_head = True
except:
    multi_head = False

obs = env.reset()
if actor_critic.use_action_masks:
    action_masks = env.env_method("get_available_actions")[0]
    for i in range(len(action_masks)):
        action_masks[i] = torch.tensor(action_masks[i], dtype=torch.float32).unsqueeze(0)
else:
    action_masks = None
render_func()

while True:
    with torch.no_grad():
        value, action, _, recurrent_hidden_states = actor_critic.act(
            obs, recurrent_hidden_states, masks, deterministic=args.det, action_masks=action_masks)

    # Obser reward and next obs
    obs, reward, done, info = env.step(action)

    action_masks_upd = []
    if actor_critic.use_action_masks:
        action_masks = info[0]["available_actions"]
        if multi_head:
            for mask in action_masks:
                action_masks_upd.append(torch.tensor(mask, dtype=torch.float32).unsqueeze(0))
        else:
            action_masks_upd = torch.tensor(action_masks, dtype=torch.float32).unsqueeze(0)
    else:
        action_masks_upd = None
    action_masks = action_masks_upd


    masks.fill_(0.0 if done else 1.0)

    render_func()
    time.sleep(args.delay)