import torch

torch.manual_seed(10)

from multi_action_head_testbed.model import MultiActionHeads

n_action_types = 3
n_heads = 4
input_dim = 10
batch_size = 8

head_infos = [
    {"type": "categorical", "out_dim": 3},
    {"type": "categorical", "out_dim": 4},
    {"type": "categorical", "out_dim": 5},
    {"type": "categorical", "out_dim": 2}
]

autoregressive_maps = [
    [-1],
    [-1, 0],
    [-1, 0, 1],
    [-1, 0]
]

action_type_masks = torch.tensor(
    [
        [1, 1, 0],
        [1, 1, 1],
        [0, 0, 1]
    ], dtype=torch.float32
)

extra_dims = [3, 8, 6]

multi_head = MultiActionHeads(head_infos=head_infos, autoregressive_maps=autoregressive_maps,
                              action_type_masks=action_type_masks, input_dim=input_dim,
                              extra_dims=extra_dims)

input = torch.randn(batch_size, input_dim)
masks = [(torch.rand(batch_size, 3) > 0.2).float(),
         (torch.rand(batch_size, 4) > 0.2).float(),
         (torch.rand(batch_size, 5) > 0.2).float(),
         (torch.rand(batch_size, 2) > 0.2).float()]

extra_inputs = [torch.randn(batch_size, 3), torch.randn(batch_size, 8), torch.randn(batch_size, 6)]

actions, joint_action_log_prob = multi_head(input, masks, extra_inputs=extra_inputs)

_, joint_action_log_prob_2 = multi_head(input, masks, extra_inputs=extra_inputs, actions=actions)

print("ok")