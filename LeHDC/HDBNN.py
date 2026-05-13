import math
from typing import Tuple, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch import Tensor
from torch.nn.parameter import Parameter


class BinLinear(nn.Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(BinLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        # self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        self.weight = Parameter((torch.rand((out_features, in_features)) * 2 - 1) * 0.001, requires_grad=True)
        # self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, input: Tensor) -> Tuple[Tensor, Tensor]:
        real_weights = self.weight
        scaling_factor = torch.mean(abs(real_weights))
        scaling_factor = scaling_factor.detach()
        binary_weights_no_grad = scaling_factor * torch.sign(real_weights)
        clipped_weights = torch.clamp(real_weights, -1.0, 1.0)
        bin_weight = binary_weights_no_grad.detach() - clipped_weights.detach() + clipped_weights
        associate_memory = torch.sign(bin_weight).detach()

        return F.linear(input, bin_weight, bias=None), associate_memory


class HDBNN(nn.Module):
    def __init__(self, input_dim, num_classes, dropout=0.0):
        super(HDBNN, self).__init__()

        self.input_dim = input_dim
        self.num_classes = num_classes

        self.bnn_classifier = BinLinear(input_dim, num_classes)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, input):
        query_vector = input
        x = self.dropout(input)
        x, associate_memory = self.bnn_classifier(x)

        return x, query_vector, associate_memory
