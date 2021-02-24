import os
import time
from collections import deque

import numpy as np
import torch

from core.arguments import get_args
from core.envs import make_vec_envs
from core.model import Policy, MultiHeadPolicy
from core.multi_action_heads import MultiActionHeads
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
                         args.gamma, args.log_dir, device, False, no_obs_norm=args.no_obs_norm)

    if args.multi_action_head:
        head_infos = envs.get_attr("head_infos")[0]
        autoregressive_maps = envs.get_attr("autoregressive_maps")[0]
        action_type_masks = torch.tensor(envs.get_attr("action_type_masks")[0], dtype=torch.float32, device=device)
        action_heads = MultiActionHeads(head_infos, autoregressive_maps, action_type_masks,
                                        input_dim=args.hidden_size)
        actor_critic = MultiHeadPolicy(envs.observation_space.shape, action_heads, use_action_masks=args.use_action_masks,
                            base_kwargs={'recurrent': args.recurrent_policy, 'recurrent_type': args.recurrent_type,
                                         'hidden_size': args.hidden_size})
    else:
        actor_critic = Policy(envs.observation_space.shape, envs.action_space, use_action_masks=args.use_action_masks,
                          base_kwargs={'recurrent': args.recurrent_policy, 'recurrent_type': args.recurrent_type,
                                       'hidden_size': args.hidden_size})
    actor_critic.to(device)

    agent = PPO(actor_critic, args.clip_param, args.ppo_epoch, args.num_mini_batch, args.value_loss_coef,
                args.entropy_coef, lr=args.lr, eps=args.eps, max_grad_norm=args.max_grad_norm,
                recompute_returns=args.recompute_returns, use_gae=args.use_gae, gamma=args.gamma,
                gae_lambda=args.gae_lambda)

    if args.multi_action_head:
        action_head_info = envs.get_attr("head_infos")[0]
    else:
        action_head_info = None
    rollouts = RolloutStorage(args.num_steps, args.num_processes, envs.observation_space.shape,
                              action_head_info = action_head_info, action_space=envs.action_space,
                              recurrent_hidden_state_size=actor_critic.recurrent_hidden_state_size,
                              multi_action_head=args.multi_action_head)

    obs = envs.reset()
    if actor_critic.use_action_masks:
        action_masks = envs.env_method("get_available_actions") #build in zip so it returns [head_1(all_envs), head_2(all_envs), ...]
        if args.multi_action_head:
            action_masks = list(zip(*action_masks))
            for i in range(len(rollouts.actions)):
                rollouts.action_masks[i][0].copy_(torch.tensor(action_masks[i]))
        else:
            rollouts.action_masks[0].copy_(torch.tensor(action_masks, dtype=torch.float32, device=device))
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
                if args.multi_action_head:
                    action_masks = [rollouts.action_masks[i][step] for i in range(len(rollouts.actions))]
                else:
                    action_masks = rollouts.action_masks[step]
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                    rollouts.obs[step], recurrent_hidden_state_in, rollouts.masks[step],
                    action_masks=action_masks
                )

            obs, reward, done, infos = envs.step(action)

            action_masks_info = []
            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])
                if actor_critic.use_action_masks:
                    action_masks_info.append(info["available_actions"])

            if actor_critic.use_action_masks:
                if args.multi_action_head:
                    action_masks = list(zip(*action_masks_info))
                    for i in range(len(action_masks)):
                        action_masks[i] = torch.tensor(action_masks[i], dtype=torch.float32, device=device)
                else:
                    action_masks = torch.tensor(action_masks_info, dtype=torch.float32, device=device)
            else:
                action_masks = None

            masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])

            rollouts.insert(obs, recurrent_hidden_states, action,
                            action_log_prob, value, reward, masks, action_masks=action_masks)

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
            # x = tuple(actor_critic.dist.logstd._bias.squeeze().detach().cpu().numpy())
            # print(("action std's: ["+', '.join(['%.2f']*len(x))+"]") % tuple([np.exp(a) for a in x]))

        if (args.eval_interval is not None and len(episode_rewards) > 1
                and j % args.eval_interval == 0):
            if args.no_obs_norm == False:
                obs_rms = utils.get_vec_normalize(envs).obs_rms
            else:
                obs_rms = None
            evaluate(actor_critic, obs_rms, args.env_name, args.seed, args.num_processes, eval_log_dir, device)

if __name__ == "__main__":
    main()