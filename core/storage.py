import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

from core.utils import _flatten_helper


class RolloutStorage:
    def __init__(self, num_steps, num_processes, obs_shape, action_head_info, recurrent_hidden_state_size,
                 action_space=None, multi_action_head=True):
        self.multi_action_head = multi_action_head
        self.obs = torch.zeros(num_steps + 1, num_processes, *obs_shape)
        self.recurrent_hidden_states = torch.zeros(num_steps+1, num_processes, recurrent_hidden_state_size)
        self.recurrent_cell_states = torch.zeros(num_steps+1, num_processes, recurrent_hidden_state_size)
        self.rewards = torch.zeros(num_steps, num_processes, 1)
        self.value_preds = torch.zeros(num_steps + 1, num_processes, 1)
        self.returns = torch.zeros(num_steps + 1, num_processes, 1)
        self.action_log_probs = torch.zeros(num_steps, num_processes, 1)
        self.actions = []
        self.action_masks = []
        if multi_action_head:
            for info in action_head_info:
                if info["type"] == "categorical":
                    action_shape = 1
                elif info["type"] == "normal":
                    action_shape = info["out_dim"]
                else:
                    raise NotImplementedError
                self.actions.append(torch.zeros(num_steps, num_processes, action_shape))
                if info["type"] == "categorical":
                    self.actions[-1] = self.actions[-1].long()
                mask_shape = info["out_dim"]
                self.action_masks.append(torch.ones(num_steps+1, num_processes, mask_shape))
        else:
            if action_space.__class__.__name__ == 'Discrete':
                action_shape = 1
            else:
                action_shape = action_space.shape[0]
            self.actions = torch.zeros(num_steps, num_processes, action_shape)
            mask_shape = action_space
            if action_space.__class__.__name__ == 'Discrete':
                self.actions = self.actions.long()
                mask_shape = action_space.n
            self.action_masks = torch.ones(num_steps+1, num_processes, mask_shape)

        self.masks = torch.ones(num_steps + 1, num_processes, 1)

        self.num_steps = num_steps
        self.step = 0

    def to(self, device):
        self.obs = self.obs.to(device)
        self.recurrent_hidden_states = self.recurrent_hidden_states.to(device)
        self.recurrent_cell_states = self.recurrent_cell_states.to(device)
        self.rewards = self.rewards.to(device)
        self.value_preds = self.value_preds.to(device)
        self.returns = self.returns.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        if self.multi_action_head:
            for i in range(len(self.actions)):
                self.actions[i] = self.actions[i].to(device)
                self.action_masks[i] = self.action_masks[i].to(device)
        else:
            self.actions = self.actions.to(device)
            self.action_masks = self.action_masks.to(device)
        self.masks = self.masks.to(device)

    def insert(self, obs, recurrent_hidden_states, actions, action_log_probs, value_preds, rewards, masks,
               action_masks=None):
        self.obs[self.step + 1].copy_(obs)
        if isinstance(recurrent_hidden_states, tuple):
            self.recurrent_cell_states[self.step + 1].copy_(recurrent_hidden_states[1])
            recurrent_hidden_states = recurrent_hidden_states[0]
        self.recurrent_hidden_states[self.step + 1].copy_(recurrent_hidden_states)
        if self.multi_action_head:
            for i in range(len(actions)):
                self.actions[i][self.step].copy_(actions[i])
                if action_masks is not None:
                    self.action_masks[i][self.step+1].copy_(action_masks[i])
        else:
            self.actions[self.step].copy_(actions)
            if action_masks is not None:
                self.action_masks[self.step+1].copy_(action_masks)
        self.action_log_probs[self.step].copy_(action_log_probs)
        self.value_preds[self.step].copy_(value_preds)
        self.rewards[self.step].copy_(rewards)
        self.masks[self.step + 1].copy_(masks)

        self.step = (self.step + 1) % self.num_steps

    def after_update(self):
        self.obs[0].copy_(self.obs[-1])
        if self.multi_action_head:
            for i in range(len(self.actions)):
                self.action_masks[i][0].copy_(self.action_masks[i][-1])
        else:
            self.action_masks[0].copy_(self.action_masks[-1])
        self.recurrent_hidden_states[0].copy_(self.recurrent_hidden_states[-1])
        self.recurrent_cell_states[0].copy_(self.recurrent_cell_states[-1])
        self.masks[0].copy_(self.masks[-1])

    def compute_returns(self, next_value, use_gae, gamma, gae_lambda):
        if use_gae:
            self.value_preds[-1] = next_value
            gae = 0
            for step in reversed(range(self.rewards.size(0))):
                delta = self.rewards[step] + gamma * self.value_preds[step + 1] * self.masks[step + 1] \
                        - self.value_preds[step]
                gae = delta + gamma * gae_lambda * self.masks[step + 1] * gae
                self.returns[step] = gae + self.value_preds[step]
        else:
            self.returns[-1] = next_value
            for step in reversed(range(self.rewards.size(0))):
                self.returns[step] = self.returns[step + 1] * gamma * self.masks[step + 1] + self.rewards[step]

    def feed_forward_generator(self, advantages, num_mini_batch=None, mini_batch_size=None):
        num_steps, num_processes = self.rewards.size()[0:2]
        batch_size = num_steps * num_processes

        if mini_batch_size is None:
            assert batch_size >= num_mini_batch, ("PPO requires number of processes ({}) * number of steps ({}) = {} to be"
                                                  "greater than or equal to number of PPO mini-batches ({}).".format(
                num_processes, num_steps, num_processes * num_steps, num_mini_batch
            ))
            mini_batch_size = batch_size // num_mini_batch

        sampler = BatchSampler(SubsetRandomSampler(range(batch_size)), mini_batch_size, drop_last=True)

        for indices in sampler:
            obs_batch = self.obs[:-1].view(-1, *self.obs.size()[2:])[indices]
            recurrent_hidden_states_batch = self.recurrent_hidden_states[:-1].view(
                -1, self.recurrent_hidden_states.size(-1)
            )[indices]
            if self.multi_action_head:
                actions_batch = [self.actions[i].view(-1, self.actions[i].size(-1))[indices]
                                 for i in range(len(self.actions))]
                actions_mask_batch = [self.action_masks[i][:-1].view(-1, self.action_masks[i].size(-1))[indices]
                                      for i in range(len(self.action_masks))]
            else:
                actions_batch = self.actions.view(-1, self.actions.size(-1))[indices]
                actions_mask_batch = self.action_masks[:-1].view(-1, self.action_masks.size(-1))[indices]
            value_preds_batch = self.value_preds[:-1].view(-1, 1)[indices]
            return_batch = self.returns[:-1].view(-1, 1)[indices]
            masks_batch = self.masks[:-1].view(-1, 1)[indices]
            old_action_log_probs_batch = self.action_log_probs.view(-1, 1)[indices]

            if advantages is None:
                adv_targ = None
            else:
                adv_targ = advantages.view(-1, 1)[indices]

            yield obs_batch, recurrent_hidden_states_batch, actions_batch, actions_mask_batch, \
                value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, adv_targ

    def recurrent_generator(self, advantages, num_mini_batch, type="GRU"):
        num_processes = self.rewards.size(1)
        assert num_processes >= num_mini_batch, (
            "PPO requires the number of processes ({}) "
            "to be greater than or equal to the number of "
            "PPO mini batches ({}).".format(num_processes, num_mini_batch))
        num_envs_per_batch = num_processes // num_mini_batch
        perm = torch.randperm(num_processes)
        for start_ind in range(0, num_processes, num_envs_per_batch):
            obs_batch = []
            recurrent_hidden_states_batch = []
            recurrent_cell_states_batch = []
            if self.multi_action_head:
                actions_batch = [[] for _ in range(len(self.actions))]
                action_masks_batch = [[] for _ in range(len(self.actions))]
            else:
                actions_batch = []
                action_masks_batch = []
            value_preds_batch = []
            return_batch = []
            masks_batch = []
            old_action_log_probs_batch = []
            adv_targ = []

            for offset in range(num_envs_per_batch):
                ind = perm[start_ind + offset]
                obs_batch.append(self.obs[:-1, ind])
                recurrent_hidden_states_batch.append(self.recurrent_hidden_states[0:1, ind])
                recurrent_cell_states_batch.append(self.recurrent_cell_states[0:1, ind])
                if self.multi_action_head:
                    for i in range(len(self.actions)):
                        actions_batch[i].append(self.actions[i][:, ind])
                        action_masks_batch[i].append(self.action_masks[i][:-1, ind])
                else:
                    actions_batch.append(self.actions[:, ind])
                    action_masks_batch.append(self.action_masks[:-1, ind])
                value_preds_batch.append(self.value_preds[:-1, ind])
                return_batch.append(self.returns[:-1, ind])
                masks_batch.append(self.masks[:-1, ind])
                old_action_log_probs_batch.append(self.action_log_probs[:, ind])
                adv_targ.append(advantages[:, ind])

            T, N = self.num_steps, num_envs_per_batch
            # These are all tensors of size (T, N, -1)
            obs_batch = torch.stack(obs_batch, 1)
            if self.multi_action_head:
                for i in range(len(self.actions)):
                    actions_batch[i] = torch.stack(actions_batch[i], 1)
                    action_masks_batch[i] = torch.stack(action_masks_batch[i], 1)
            else:
                actions_batch = torch.stack(actions_batch, 1)
                action_masks_batch = torch.stack(action_masks_batch, 1)
            value_preds_batch = torch.stack(value_preds_batch, 1)
            return_batch = torch.stack(return_batch, 1)
            masks_batch = torch.stack(masks_batch, 1)
            old_action_log_probs_batch = torch.stack(old_action_log_probs_batch, 1)
            adv_targ = torch.stack(adv_targ, 1)
            # States is just a (N, -1) tensor
            recurrent_hidden_states_batch = torch.stack(recurrent_hidden_states_batch, 1).view(N, -1)
            recurrent_cell_states_batch = torch.stack(recurrent_cell_states_batch, 1).view(N, -1)

            # Flatten the (T, N, ...) tensors to (T * N, ...)
            obs_batch = _flatten_helper(T, N, obs_batch)
            if self.multi_action_head:
                for i in range(len(self.actions)):
                    actions_batch[i] = _flatten_helper(T, N, actions_batch[i])
                    action_masks_batch[i] = _flatten_helper(T, N, action_masks_batch[i])
            else:
                actions_batch = _flatten_helper(T, N, actions_batch)
                action_masks_batch = _flatten_helper(T, N, action_masks_batch)
            value_preds_batch = _flatten_helper(T, N, value_preds_batch)
            return_batch = _flatten_helper(T, N, return_batch)
            masks_batch = _flatten_helper(T, N, masks_batch)
            old_action_log_probs_batch = _flatten_helper(T, N, old_action_log_probs_batch)
            adv_targ = _flatten_helper(T, N, adv_targ)

            if type == "LSTM":
                recurrent_hidden_states_batch = (recurrent_hidden_states_batch, recurrent_cell_states_batch)

            yield obs_batch, recurrent_hidden_states_batch, actions_batch, action_masks_batch, \
                  value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, adv_targ