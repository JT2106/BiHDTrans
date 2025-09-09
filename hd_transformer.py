import math
import warnings
from typing import Optional, Any, Union, Callable, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torchhd as hd
from torch import Tensor, autograd
from torch.nn.parameter import Parameter, UninitializedParameter
from torchhd import embeddings, VSATensor


class BindingLayer(nn.Module):
    __constants__ = 'dimension'
    dimension: int
    weight: Tensor

    def __init__(self, dimension: int, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(BindingLayer, self).__init__()
        self.dimension = dimension
        self.weight = Parameter((torch.rand(dimension) * 2 - 1) * 0.001, requires_grad=True)
        # self.weight = Parameter(hd.empty(1, dimension, requires_grad=True, **factory_kwargs))
        # self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, input: VSATensor) -> Tensor:
        real_weights = self.weight
        # binary_weights_no_grad = hd.hard_quantize(real_weights)
        binary_weights_no_grad = torch.sign(real_weights)
        binary_weights = binary_weights_no_grad.detach() - real_weights.detach() + real_weights
        binary_weights = binary_weights.to(input.device)

        return hd.bind(input, binary_weights)


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
        self.weight = Parameter((torch.rand((out_features, in_features)) * 2 - 1) * 0.001, requires_grad=True)
        # self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs))
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


class HDMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout, kdim=None, vdim=None,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(HDMultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout_p = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.binding_layer_q = BindingLayer(dimension=embed_dim)
        self.binding_layer_k = BindingLayer(dimension=embed_dim)
        self.binding_layer_v = BindingLayer(dimension=embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)
        self.binding_attn_output = BindingLayer(dimension=embed_dim)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, key_padding_mask: Optional[Tensor] = None,
                need_weights: bool = True, attn_mask: Optional[Tensor] = None,
                average_attn_weights: bool = True) -> Tuple[Tensor, Optional[Tensor]]:

        attn_output, attn_output_weights = self.hd_multi_head_attention_forward(
            query, key, value, self.embed_dim, self.num_heads, dropout_p=self.dropout_p,
            training=self.training, key_padding_mask=key_padding_mask, need_weights=need_weights,
            attn_mask=attn_mask, average_attn_weights=average_attn_weights)

        return attn_output, attn_output_weights

    def hd_multi_head_attention_forward(self, query: Tensor, key: Tensor, value: Tensor, embed_dim_to_check: int,
                                        num_heads: int, dropout_p: float, training: bool = True,
                                        key_padding_mask: Optional[Tensor] = None, need_weights: bool = True,
                                        attn_mask: Optional[Tensor] = None, average_attn_weights: bool = True
                                        ) -> Tuple[Tensor, Optional[Tensor]]:
        # set up shape vars
        tgt_len, bsz, embed_dim = query.shape
        src_len, _, _ = key.shape
        assert embed_dim == embed_dim_to_check, \
            f"was expecting embedding dimension of {embed_dim_to_check}, but got {embed_dim}"
        if isinstance(embed_dim, torch.Tensor):
            # embed_dim can be a tensor when JIT tracing
            head_dim = embed_dim.div(num_heads, rounding_mode='trunc')
        else:
            head_dim = embed_dim // num_heads
        assert head_dim * num_heads == embed_dim, f"embed_dim {embed_dim} not divisible by num_heads {num_heads}"

        #
        # compute in-projection
        #
        q, k, v = self._hd_in_projection_packed(query, key, value)

        # prep attention mask
        if attn_mask is not None:
            if attn_mask.dtype == torch.uint8:
                warnings.warn(
                    "Byte tensor for attn_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.")
                attn_mask = attn_mask.to(torch.bool)
            else:
                assert attn_mask.is_floating_point() or attn_mask.dtype == torch.bool, \
                    f"Only float, byte, and bool types are supported for attn_mask, not {attn_mask.dtype}"
            # ensure attn_mask's dim is 3
            if attn_mask.dim() == 2:
                correct_2d_size = (tgt_len, src_len)
                if attn_mask.shape != correct_2d_size:
                    raise RuntimeError(
                        f"The shape of the 2D attn_mask is {attn_mask.shape}, but should be {correct_2d_size}.")
                attn_mask = attn_mask.unsqueeze(0)
            elif attn_mask.dim() == 3:
                correct_3d_size = (bsz * num_heads, tgt_len, src_len)
                if attn_mask.shape != correct_3d_size:
                    raise RuntimeError(
                        f"The shape of the 3D attn_mask is {attn_mask.shape}, but should be {correct_3d_size}.")
            else:
                raise RuntimeError(f"attn_mask's dimension {attn_mask.dim()} is not supported")

        # prep key padding mask
        if key_padding_mask is not None and key_padding_mask.dtype == torch.uint8:
            warnings.warn(
                "Byte tensor for key_padding_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.")
            key_padding_mask = key_padding_mask.to(torch.bool)

        #
        # reshape q, k, v for multihead attention and make em batch first
        #
        q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
        k = k.contiguous().view(k.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
        v = v.contiguous().view(v.shape[0], bsz * num_heads, head_dim).transpose(0, 1)

        # adjust dropout probability
        if not training:
            dropout_p = 0.0

        #
        # calculate attention and out projection
        #
        attn_output, attn_output_weights = self._hd_scaled_dot_product_attention(q, k, v, attn_mask, dropout_p)
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        binary_attn_output_no_grad = torch.sign(attn_output)
        binary_attn_output = binary_attn_output_no_grad.detach() - attn_output.detach() + attn_output

        binary_attn_output = self.binding_attn_output(binary_attn_output)

        return binary_attn_output, None

    def _hd_in_projection_packed(self, q: Tensor, k: Tensor, v: Tensor) -> Tuple[Any, Any, Any]:
        if k is v and q is k:
            # self-attention
            q_out = self.binding_layer_q(q)
            k_out = self.binding_layer_k(k)
            v_out = self.binding_layer_v(v)

            return q_out, k_out, v_out
        else:
            raise ValueError("This code do NOT support this type of attention")

    def _hd_scaled_dot_product_attention(self, q: Tensor, k: Tensor, v: Tensor,
                                         attn_mask: Optional[Tensor] = None,  dropout_p: float = 0.0
                                         ) -> Tuple[Tensor, Tensor]:
        B, Nt, E = q.shape
        q = q / math.sqrt(E)

        # (B, Nt, E) x (B, E, Ns) -> (B, Nt, Ns)
        attn = torch.bmm(q, k.transpose(-2, -1))
        if attn_mask is not None:
            attn += attn_mask
        # attn = self.softmax(attn)
        real_attn = attn
        binary_attn_no_grad = torch.where(real_attn > 0,
                                          torch.tensor(1.0).to(real_attn.device),
                                          torch.tensor(0.0).to(real_attn.device))
        attn = binary_attn_no_grad.detach() - real_attn.detach() + real_attn
        if dropout_p > 0.0:
            attn = self.dropout(attn)

        # (B, Nt, Ns) x (B, Ns, E) -> (B, Nt, E)
        output = torch.bmm(attn, v)

        return output, attn


class HDTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dropout: float,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(HDTransformerEncoderLayer, self).__init__()
        self.self_attn = HDMultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        x = src
        x = self._sa_block(x, src_mask, src_key_padding_mask)


        return x

    # self-attention block
    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False)[0]
        return self.dropout(x)


class HDTransformerModel(nn.Module):
    def __init__(self, input_dim, num_classes, seq_len, num_heads=10, dropout=0.0):
        super(HDTransformerModel, self).__init__()

        self.input_dim = input_dim
        self.num_classes = num_classes
        self.seq_len = seq_len

        self.encoder_layer = HDTransformerEncoderLayer(d_model=input_dim, nhead=num_heads, dropout=dropout)
        self.classifier = BinLinear(input_dim, num_classes)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # (batch_size, seq_len, dmodel) --> (seq_len, batch_size, dmodel)
        x = x.transpose(0, 1)

        # HD transformer
        output = self.encoder_layer(x)

        # position 0 output for classification (if applied)
        # query_vector = output[0, :, :]

        query_vector = output[-1, :, :]

        # classifier
        output = self.dropout(query_vector)
        output, associate_memory = self.classifier(output)

        return output, query_vector, associate_memory
