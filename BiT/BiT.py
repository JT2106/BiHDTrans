import math
import warnings
from typing import Optional, Any, Union, Callable, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.init import constant_, xavier_normal_, xavier_uniform_
from torch import Tensor, autograd
from torch.nn.parameter import Parameter, UninitializedParameter


class BinarizationSign(autograd.Function):
    @staticmethod
    def jvp(ctx: Any, *grad_inputs: Any) -> Any:
        pass

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        real_input = input - torch.mean(input)
        real_input = real_input.detach()
        scaling_factor = torch.mean(abs(real_input))
        scaling_factor = scaling_factor.detach()
        binary_output = scaling_factor * torch.sign(real_input)

        return binary_output

    @staticmethod
    def backward(ctx, grad_output):
        # 获取保存的输入数据
        input, = ctx.saved_tensors
        # 初始化对input的梯度
        grad_input = grad_output.clone()

        # 仅对二值化后的权重计算梯度
        # 对于 |x| > 1 的部分，梯度为 0；对于 |x| <= 1 的部分，梯度为 1
        grad_input[input.abs() > 1] = 0  # 对于 |x| > 1，梯度为0
        grad_input[input.abs() <= 1] = grad_output[input.abs() <= 1]  # 对于 |x| <= 1，梯度为grad_output

        return grad_input


class BinarizationSignLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return BinarizationSign.apply(input)


'''
class BinarizationBool(autograd.Function):
    @staticmethod
    def jvp(ctx: Any, *grad_inputs: Any) -> Any:
        pass

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        # real_input = input - torch.mean(input)
        real_input = input
        real_input = real_input.detach()
        # scaling_factor = torch.mean(abs(real_input))
        scaling_factor = torch.tensor(1)
        scaling_factor = scaling_factor.detach()
        binary_output = scaling_factor * torch.where(real_input > 0.5,
                                                     torch.tensor(1.0, device=input.device),
                                                     torch.tensor(0.0, device=input.device))

        return binary_output

    @staticmethod
    def backward(ctx, grad_output):
        # 获取保存的输入数据
        input, = ctx.saved_tensors
        # 初始化对input的梯度
        grad_input = grad_output.clone()

        # 仅对二值化后的权重计算梯度
        # 对于 |x| > 1 的部分，梯度为 0；对于 |x| <= 1 的部分，梯度为 1
        grad_input[input.abs() > 1] = 0  # 对于 |x| > 1，梯度为0
        grad_input[input.abs() <= 1] = grad_output[input.abs() <= 1]  # 对于 |x| <= 1，梯度为grad_output

        return grad_input
'''


class BinarizationBool(autograd.Function):
    @staticmethod
    def jvp(ctx: Any, *grad_inputs: Any) -> Any:
        pass

    @staticmethod
    def forward(ctx, input, alpha, beta):
        # 保存输入和参数以供反向传播使用
        ctx.save_for_backward(input, alpha, beta)

        # 计算二值化输出
        clipped_input = torch.clamp((input - beta) / alpha, 0, 1)  # 裁剪输入到 [0, 1]
        binary_output = alpha * torch.round(clipped_input)  # 四舍五入并应用缩放因子

        return binary_output

    @staticmethod
    def backward(ctx, grad_output):
        # 获取保存的输入和参数
        input, alpha, beta = ctx.saved_tensors

        # 初始化梯度
        grad_input = grad_output.clone()

        # 计算输入的梯度
        # 对于 |x| > 1 的部分，梯度为 0；对于 |x| <= 1 的部分，梯度为 grad_output
        grad_input[input.abs() > 1] = 0
        grad_input[input.abs() <= 1] = grad_output[input.abs() <= 1]

        # 计算 alpha 的梯度
        # 根据论文中的公式 (10)
        alpha_grad = torch.where(
            input < beta / 1,
            torch.tensor(0.0, device=input.device),
            torch.where(
                input < (alpha / 2 + beta),
                (beta - input) / alpha,
                torch.where(
                    input < (alpha + beta) / 1,
                    1.0 - ((input - beta) / alpha),
                    torch.tensor(1.0, device=input.device)
                )
            )
        )
        grad_alpha = (grad_output * alpha_grad).sum()

        beta_grad = torch.where(
            input < beta / 1,
            torch.tensor(0.0, device=input.device),
            torch.where(
                input < (alpha + beta) / 1,
                torch.tensor(-1.0, device=input.device),
                torch.tensor(0.0, device=input.device)
            )
        )
        grad_beta = (grad_output * beta_grad).sum()

        return grad_input, grad_alpha, grad_beta


class BinarizationBoolLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha = torch.nn.Parameter(torch.tensor(1.0))
        self.beta = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, input):
        return BinarizationBool.apply(input, self.alpha, self.beta)


class BinLinear(nn.Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = False,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(BinLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        self.register_parameter('bias', None)
        self.reset_parameters()
        self.bin_sign = BinarizationSignLayer()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, input: Tensor) -> Tensor:
        bin_weight = self.bin_sign(self.weight).to(input.device, non_blocking=True)
        return F.linear(input, bin_weight, bias=None)


class BinMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout, kdim=None, vdim=None,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(BinMultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout_p = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.bin_linear_q = BinLinear(in_features=embed_dim, out_features=embed_dim)
        self.bin_linear_k = BinLinear(in_features=embed_dim, out_features=embed_dim)
        self.bin_linear_v = BinLinear(in_features=embed_dim, out_features=embed_dim)
        self.bin_sign_q = BinarizationSignLayer()
        self.bin_sign_k = BinarizationSignLayer()
        self.bin_sign_v = BinarizationSignLayer()
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)
        self.bin_bool_attnscore = BinarizationBoolLayer()
        self.linear_attn_output = BinLinear(in_features=embed_dim, out_features=embed_dim)
        self.bin_sign_attnout = BinarizationSignLayer()

    def forward(self, query: Tensor, key: Tensor, value: Tensor, key_padding_mask: Optional[Tensor] = None,
                need_weights: bool = True, attn_mask: Optional[Tensor] = None,
                average_attn_weights: bool = True) -> Tuple[Tensor, Optional[Tensor]]:

        attn_output, attn_output_weights = self._multi_head_attention_forward(
            query, key, value, self.embed_dim, self.num_heads, dropout_p=self.dropout_p,
            training=self.training, key_padding_mask=key_padding_mask, need_weights=need_weights,
            attn_mask=attn_mask, average_attn_weights=average_attn_weights)

        return attn_output, attn_output_weights

    def _multi_head_attention_forward(self, query: Tensor, key: Tensor, value: Tensor, embed_dim_to_check: int,
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
        q, k, v = self._bin_in_projection_packed(query, key, value)

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
        attn_output, attn_output_weights = self._bin_scaled_dot_product_attention(q, k, v, attn_mask, dropout_p)
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        binary_attn_output = self.bin_sign_attnout(attn_output)

        binary_attn_output = self.linear_attn_output(binary_attn_output)

        return binary_attn_output, None

    def _bin_in_projection_packed(self, q: Tensor, k: Tensor, v: Tensor) -> Tuple[Any, Any, Any]:
        if k is v and q is k:
            # self-attention
            q_out = self.bin_linear_q(q)
            k_out = self.bin_linear_k(k)
            v_out = self.bin_linear_v(v)

            q_out = self.bin_sign_q(q_out)
            k_out = self.bin_sign_k(k_out)
            v_out = self.bin_sign_v(v_out)

            return q_out, k_out, v_out
        else:
            raise ValueError("This code do NOT support this type of attention")

    def _bin_scaled_dot_product_attention(self, q: Tensor, k: Tensor, v: Tensor,
                                          attn_mask: Optional[Tensor] = None, dropout_p: float = 0.0
                                          ) -> Tuple[Tensor, Tensor]:
        B, Nt, E = q.shape
        q = q / math.sqrt(E)

        # (B, Nt, E) x (B, E, Ns) -> (B, Nt, Ns)
        attn = torch.bmm(q, k.transpose(-2, -1))
        if attn_mask is not None:
            attn += attn_mask
        attn = self.softmax(attn)
        bin_attn = self.bin_bool_attnscore(attn)
        if dropout_p > 0.0:
            bin_attn = self.dropout(bin_attn)

        # (B, Nt, Ns) x (B, Ns, E) -> (B, Nt, E)
        output = torch.bmm(bin_attn, v)

        return output, bin_attn


class BinTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(BinTransformerEncoderLayer, self).__init__()
        self.self_attn = BinMultiheadAttention(d_model, nhead, dropout=dropout, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.linear1 = BinLinear(d_model, dim_feedforward, **factory_kwargs)
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        self.linear2 = BinLinear(dim_feedforward, d_model, **factory_kwargs)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.bin_sign1 = BinarizationSignLayer()
        self.bin_sign2 = BinarizationSignLayer()
        self.bin_bool = BinarizationBoolLayer()

        self.alpha = torch.tensor(1.0, requires_grad=True)
        self.beta = torch.tensor(0.0, requires_grad=True)

        self.alpha = Parameter(torch.tensor(1.0, requires_grad=True, **factory_kwargs))

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        x = self.bin_sign1(src)
        x = x + self._sa_block(x, src_mask, src_key_padding_mask)
        x = self.norm1(x)
        x = x + self._ff_block(x)
        x = self.norm2(x)

        return x

    # self-attention block
    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False)[0]
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.bin_sign2(x)
        x = self.activation(self.linear1(x))
        x = self.bin_bool(x)
        x = self.linear2(self.dropout(x))
        return self.dropout2(x)


class BiTModel(nn.Module):
    def __init__(self, input_dim, num_classes, seq_len,
                 dmodel=128, num_heads=2, hidden_dim=256, dropout=0.1):
        super(BiTModel, self).__init__()

        self.input_dim = input_dim
        self.num_classes = num_classes
        self.seq_len = seq_len
        self.dmodel = dmodel

        self.input_projection = BinLinear(input_dim, dmodel)
        self.transformer_encoder = BinTransformerEncoderLayer(d_model=dmodel, nhead=num_heads,
                                                              dim_feedforward=hidden_dim,
                                                              dropout=dropout)
        self.bin_sign = BinarizationSignLayer()
        self.classifier = BinLinear(dmodel, num_classes)

    def forward(self, x):
        # 输入x的维度为 (batch_size, seq_len, input_dim)
        # 使用线性变换将 input_dim 映射到 dmodel
        x = self.input_projection(x)
        # 需要调整为 (seq_len, batch_size, dmodel) 以适配 Transformer 输入
        x = x.permute(1, 0, 2)

        # Transformer编码器处理
        output = self.transformer_encoder(x)
        output = self.bin_sign(output)

        # 使用最后一个时间步的输出进行分类
        # 选择输出序列中的最后一个时间步的数据进行分类
        output = output[-1, :, :]

        # 通过全连接层进行分类
        output = self.classifier(output)

        return output
