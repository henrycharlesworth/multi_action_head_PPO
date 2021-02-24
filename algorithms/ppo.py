import torch
import torch.nn as nn
import torch.optim as optim

from core.utils import _flatten_helper

class PPO():
    def __init__(self, actor_critic, clip_param, ppo_epoch, num_mini_batch, value_loss_coef, entropy_coef,
                 lr=None, eps=None, max_grad_norm=None, use_clipped_value_loss=True, recompute_returns=False,
                 use_gae=True, gamma=0.99, gae_lambda=0.95):
        self.actor_critic = actor_critic

        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.recompute_returns = recompute_returns
        self.use_gae = use_gae
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        self.optimizer = optim.Adam(actor_critic.parameters(), lr=lr, eps=eps)

    def update(self, rollouts):
        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0

        advantages = None

        for e in range(self.ppo_epoch):
            if e == 0 or (e > 0 and self.recompute_returns):
                with torch.no_grad():
                    if e > 0:
                        #recomputing advantages at start of each ppo epoch.
                        if self.actor_critic.is_recurrent:
                            T, N = rollouts.obs[:-1].shape[0], rollouts.obs[:-1].shape[1]
                            if self.actor_critic.base.recurrent_type == "LSTM":
                                recurrent_hidden_states_in = (_flatten_helper(T, N, rollouts.recurrent_hidden_states[:-1]),
                                                              _flatten_helper(T, N, rollouts.recurrent_cell_states[:-1]))
                            else:
                                recurrent_hidden_states_in = _flatten_helper(T, N, rollouts.recurrent_cell_states[:-1])
                            obs_in = _flatten_helper(T, N, rollouts.obs[:-1])
                            mask_in = _flatten_helper(T, N, rollouts.masks[:-1])
                        else:
                            recurrent_hidden_states_in = rollouts.recurrent_hidden_states[:-1]
                            obs_in = rollouts.obs[:-1]
                            mask_in = rollouts.masks[:-1]
                        values = self.actor_critic.get_value(
                            obs_in, recurrent_hidden_states_in, mask_in
                        ).detach()
                        if self.actor_critic.is_recurrent:
                            values = values.view(T, N, -1)
                        rollouts.value_preds[:-1].copy_(values)
                    if self.actor_critic.is_recurrent and self.actor_critic.base.recurrent_type == "LSTM":
                        recurrent_hidden_states_in = (rollouts.recurrent_hidden_states[-1],
                                                      rollouts.recurrent_cell_states[-1])
                    else:
                        recurrent_hidden_states_in = rollouts.recurrent_hidden_states[-1]
                    next_value = self.actor_critic.get_value(
                        rollouts.obs[-1], recurrent_hidden_states_in, rollouts.masks[-1]
                    ).detach()
                rollouts.compute_returns(next_value, self.use_gae, self.gamma, self.gae_lambda)
                advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

            if self.actor_critic.is_recurrent:
                data_generator = rollouts.recurrent_generator(advantages, self.num_mini_batch,
                                                              type=self.actor_critic.base.recurrent_type)
            else:
                data_generator = rollouts.feed_forward_generator(advantages, self.num_mini_batch)

            for sample in data_generator:
                obs_batch, recurrent_hidden_states_batch, actions_batch, action_masks_batch, value_preds_batch, \
                return_batch, masks_batch, old_action_log_probs_batch, adv_targ = sample

                #single forward pass for all steps

                values, action_log_probs, dist_entropy, _ = self.actor_critic.evaluate_actions(
                    obs_batch, recurrent_hidden_states_batch, masks_batch, actions_batch,
                    action_masks=action_masks_batch
                )

                ratio = torch.exp(action_log_probs - old_action_log_probs_batch)
                surr1 = ratio * adv_targ
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ
                action_loss = -torch.min(surr1, surr2).mean()

                if self.use_clipped_value_loss:
                    value_pred_clipped = value_preds_batch + \
                                         (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
                    value_losses = (values - return_batch).pow(2)
                    value_losses_clipped = (value_pred_clipped - return_batch).pow(2)
                    value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = 0.5 * (return_batch - values).pow(2).mean()

                self.optimizer.zero_grad()
                (value_loss * self.value_loss_coef + action_loss - dist_entropy * self.entropy_coef).backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()

        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch