import torch
import torch.nn as nn

from core.utils import AddBias, init

"""
Modifying standard PyTorch distributions for compatibility
"""

# #differentiable categorical
# class FixedExpRelaxedCategorical(torch.distributions.relaxed_categorical.ExpRelaxedCategorical):
#     def sample(self, sample_shape=torch.Size()):
#         return super().rsample(sample_shape=sample_shape).unsqueeze(-1)
#
#     def log_prob(self, actions):
#         return super().log_prob(actions.squeeze(-1)).view(actions.size(0), -1).sum(-1).unsqueeze(-1)
#
# class RelaxedCategorical(nn.Module):
#     def __init__(self, num_inputs, num_outputs):
#         super(RelaxedCategorical, self).__init__()
#
#         init_ = lambda m: init(
#             m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=0.01
#         )
#
#         self.linear = init_(nn.Linear(num_inputs, num_outputs))
#
#     def forward(self, x, mask=None):
#         x = self.linear(x)
#         if mask is not None:
#             return FixedExpRelaxedCategorical(logits=x + torch.log(mask))
#         else:
#             return FixedExpRelaxedCategorical(logits=x)

#Categorical
class FixedCategorical(torch.distributions.Categorical):
    def sample(self):
        return super().sample().unsqueeze(-1)

    def log_probs(self, actions):
        return super().log_prob(actions.squeeze(-1)).view(actions.size(0), -1).sum(-1).unsqueeze(-1)

class Categorical(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(Categorical, self).__init__()

        init_ = lambda m: init(
            m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=0.01
        )

        self.linear = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x, mask=None):
        x = self.linear(x)
        if mask is not None:
            return FixedCategorical(logits=x + torch.log(mask))
        else:
            return FixedCategorical(logits=x)

#Normal
class FixedNormal(torch.distributions.Normal):
    def log_probs(self, actions):
        return super().log_prob(actions).sum(-1, keepdim=True)

    def entropy(self):
        return super().entropy().sum(-1)

    def mode(self):
        return self.mean

class DiagGaussian(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(DiagGaussian, self).__init__()

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))

        self.fc_mean = init_(nn.Linear(num_inputs, num_outputs))
        desired_init_log_std = -0.693471
        self.logstd = AddBias(desired_init_log_std * torch.ones(num_outputs)) #so no state-dependent sigma
        # self.logstd = AddBias(torch.zeros(num_outputs))

    def forward(self, x):
        action_mean = self.fc_mean(x)

        zeros = torch.zeros(action_mean.size())
        if x.is_cuda:
            zeros = zeros.cuda()

        action_logstd = self.logstd(zeros)
        return FixedNormal(action_mean, action_logstd.exp())
