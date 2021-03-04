import torch
import torch.nn as nn

from core.distributions import Categorical, DiagGaussian

class MultiActionHeads(nn.Module):
    def __init__(self, head_infos, autoregressive_maps, action_type_masks, input_dim, action_heads=None,
                 extra_dims=None):
        """
        head_info - list of dicts, with type (continuous/discrete) and dimension. first entry assumed to be action type.
        autoregressive map - list of adjacency lists that detail which action heads feed into which later heads (-1 is input state)
        action_type_mask - list of action_type masks for each action head (after first which is action type)
                           (each is list of len(action_types) where 1 means action_type(ind) means this action_type
                           requires this head)
        action_heads - can provide ModuleList of action heads instead (for full customisation)
        extra_dims - list for action_heads (after first one) for any extra input dimension.
        """
        super(MultiActionHeads, self).__init__()
        self.dummy_param = nn.Parameter(torch.empty(0))
        self.head_infos = head_infos
        self.autoregressive_map = autoregressive_maps
        self.action_type_mask = action_type_masks
        self.input_dim = input_dim
        self.extra_dims = extra_dims

        if action_heads is None:
            assert head_infos[0]["type"] == "categorical" #first action head is action type
            assert len(head_infos) == len(autoregressive_maps)
            assert action_type_masks.shape[0] == head_infos[0]["out_dim"]
            assert action_type_masks.shape[1] == len(head_infos) - 1
            for i, map in enumerate(autoregressive_maps):
                for entry in map:
                    assert entry < i #only allow earlier heads as inputs to later heads

            self.action_heads = nn.ModuleList()
            for i, info in enumerate(head_infos):
                type = info["type"]; head_out_dim = info["out_dim"]
                head_in_dim = 0
                for ind in autoregressive_maps[i]:
                    if ind == -1:
                        head_in_dim += input_dim
                    else:
                        head_in_dim += head_infos[ind]["out_dim"]
                if extra_dims is not None and i > 0:
                    head_in_dim += extra_dims[i - 1]
                self.action_heads.append(ActionHead(head_in_dim, head_out_dim, type))
        else:
            assert isinstance(action_heads, nn.ModuleList)
            self.action_heads = action_heads

    def forward(self, input, masks, actions=None, extra_inputs=None, deterministic=False):
        head_outputs = []
        action_outputs = []
        action_type_dist = self.action_heads[0](input, masks[0])
        if deterministic:
            action_type = action_type_dist.mode()
        else:
            action_type = action_type_dist.sample()
        one_hot_action_types = torch.zeros(input.size(0), self.head_infos[0]["out_dim"], device=self.dummy_param.device)
        if actions is None:
            one_hot_action_types.scatter_(-1, action_type, 1.0)
            type_masks = self.action_type_mask[action_type.squeeze(-1), :]
        else:
            one_hot_action_types.scatter_(-1, actions[0], 1.0)
            type_masks = self.action_type_mask[actions[0].squeeze(-1), :]
        head_outputs.append(one_hot_action_types)
        action_outputs.append(action_type)
        if actions is None:
            joint_action_log_prob = action_type_dist.log_probs(action_type)
        else: #evaluating actions rather than generating
            joint_action_log_prob = action_type_dist.log_probs(actions[0])

        entropy = action_type_dist.entropy().mean()
        for i in range(1, len(self.action_heads)):
            head_inputs = []
            for ind in self.autoregressive_map[i]:
                if ind == -1:
                    head_inputs.append(input)
                else:
                    head_inputs.append(head_outputs[ind])
            if extra_inputs is not None:
                head_inputs.append(extra_inputs[i-1])
            head_input = torch.cat(head_inputs, dim=-1)

            head_dist = self.action_heads[i](head_input, masks[i])
            if deterministic:
                head_action = head_dist.mode()
            else:
                if self.head_infos[i]["type"] == "normal":
                    head_action = head_dist.rsample()
                else:
                    head_action = head_dist.sample()
            if self.head_infos[i]["type"] == "categorical":
                one_hot_head_action = torch.zeros(input.size(0), self.head_infos[i]["out_dim"],
                                                  device=self.dummy_param.device)
                if actions is None:
                    one_hot_head_action.scatter_(-1, head_action, 1.0)
                else:
                    one_hot_head_action.scatter_(-1, actions[i], 1.0)
                head_outputs.append(one_hot_head_action)
            else:
                head_outputs.append(head_action)
            action_outputs.append(head_action)

            log_prob_mask = type_masks[:, i-1].view(-1, 1)
            if actions is None:
                joint_action_log_prob += (log_prob_mask * head_dist.log_probs(head_action))
            else:
                joint_action_log_prob += (log_prob_mask * head_dist.log_probs(actions[i]))

            entropy += head_dist.entropy().mean()

        return action_outputs, joint_action_log_prob, entropy


class ActionHead(nn.Module):
    def __init__(self, input_dim, output_dim, type="categorical"):
        super(ActionHead, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.type = type
        if type == "categorical":
            self.distribution = Categorical(num_inputs=input_dim, num_outputs=output_dim)
        elif type == "normal":
            self.distribution = DiagGaussian(num_inputs=input_dim, num_outputs=output_dim)
        else:
            raise NotImplementedError

    def forward(self, input, mask):
        if self.type == "normal":
            return self.distribution(input)
        else:
            return self.distribution(input, mask)