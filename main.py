import os
import time
from collections import deque

import numpy as np
import torch

from core.arguments import get_args
from core.envs import make_vec_envs
from core.model import Policy
from core.storage import RolloutStorage
from core import utils

from algorithms.ppo import PPO
from evaluate import evaluate


def main():
    args = get_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    log_dir = os.path.expanduser(args.log_dir)
    eval_log_dir = log_dir + "_eval"
    utils.cleanup_log_dir(log_dir)
    utils.cleanup_log_dir(eval_log_dir)

    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")

    envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
                         args.gamma, args.log_dir, device, False)

    actor_critic = Policy(envs.observation_space.shape, envs.action_space,
                          base_kwargs={'recurrent': args.recurrent_policy, 'recurrent_type': args.recurrent_type})
    actor_critic.to(device)

    agent = PPO(actor_critic, args.clip_param, args.ppo_epoch, args.num_mini_batch, args.value_loss_coef,
                args.entropy_coef, lr=args.lr, eps=args.eps, max_grad_norm=args.max_grad_norm,
                recompute_returns=args.recompute_returns, use_gae=args.use_gae, gamma=args.gamma,
                gae_lambda=args.gae_lambda)

    rollouts = RolloutStorage(args.num_steps, args.num_processes, envs.observation_space.shape, envs.action_space,
                              actor_critic.recurrent_hidden_state_size)

    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    episode_rewards = deque(maxlen=10)

    start = time.time()
    num_updates = int(args.num_env_steps) // args.num_steps // args.num_processes

    for j in range(num_updates):

        if args.use_linear_lr_decay:
            utils.update_linear_schedule(agent.optimizer, j, num_updates, args.lr)

        for step in range(args.num_steps):
            with torch.no_grad():
                if actor_critic.is_recurrent and actor_critic.base.recurrent_type == "LSTM":
                    recurrent_hidden_state_in = (rollouts.recurrent_hidden_states[step],
                                              rollouts.recurrent_cell_states[step])
                else:
                    recurrent_hidden_state_in = rollouts.recurrent_hidden_states[step]
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                    rollouts.obs[step], recurrent_hidden_state_in, rollouts.masks[step]
                )

            obs, reward, done, infos = envs.step(action)

            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])

            masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])

            rollouts.insert(obs, recurrent_hidden_states, action,
                            action_log_prob, value, reward, masks)

        value_loss, action_loss, dist_entropy = agent.update(rollouts)
        rollouts.after_update()

        # save for every interval-th episode or for the last epoch
        if (j % args.save_interval == 0 or j == num_updates - 1) and args.save_dir != "":
            save_path = os.path.join(args.save_dir, args.algo)
            try:
                os.makedirs(save_path)
            except OSError:
                pass

            torch.save([actor_critic, getattr(utils.get_vec_normalize(envs), 'obs_rms', None)],
                       os.path.join(save_path, args.env_name + args.extra_id + ".pt"))

        if j % args.log_interval == 0 and len(episode_rewards) > 1:
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            end = time.time()
            print(
                "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n"
                    .format(j, total_num_steps, int(total_num_steps / (end - start)), len(episode_rewards),
                            np.mean(episode_rewards), np.median(episode_rewards), np.min(episode_rewards),
                            np.max(episode_rewards), dist_entropy, value_loss, action_loss))
            x = tuple(actor_critic.dist.logstd._bias.squeeze().detach().cpu().numpy())
            print(("action std's: ["+', '.join(['%.2f']*len(x))+"]") % tuple([np.exp(a) for a in x]))

        if (args.eval_interval is not None and len(episode_rewards) > 1
                and j % args.eval_interval == 0):
            obs_rms = utils.get_vec_normalize(envs).obs_rms
            evaluate(actor_critic, obs_rms, args.env_name, args.seed, args.num_processes, eval_log_dir, device)

if __name__ == "__main__":
    main()