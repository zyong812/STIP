import torch
import warnings
from torch.nn.functional import linear, pad, softmax, dropout # containing original multi_head_attention_forward implementation
from torch.nn.init import xavier_normal_, xavier_uniform_, constant_
from torch.nn.modules.activation import Parameter # containing original MultiheadAttention implementation
from torch.nn.modules.linear import _LinearWithBias
import torch.nn.functional as F
import numpy as np
# visualize results
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as PathEffects
from src.util import box_ops


Tensor = torch.Tensor

def multi_head_attention_forward_with_role(query,                 # type: Tensor
                                 key,                             # type: Tensor
                                 value,                           # type: Tensor
                                 embed_dim_to_check,              # type: int
                                 num_heads,                       # type: int
                                 in_proj_weight,                  # type: Tensor
                                 in_proj_bias,                    # type: Tensor
                                 bias_k,                          # type: Optional[Tensor]
                                 bias_v,                          # type: Optional[Tensor]
                                 add_zero_attn,                   # type: bool
                                 dropout_p,                       # type: float
                                 out_proj_weight,                 # type: Tensor
                                 out_proj_bias,                   # type: Tensor
                                 training=True,                   # type: bool
                                 key_padding_mask=None,           # type: Optional[Tensor]
                                 need_weights=True,               # type: bool
                                 attn_mask=None,                  # type: Optional[Tensor]
                                 use_separate_proj_weight=False,  # type: bool
                                 q_proj_weight=None,              # type: Optional[Tensor]
                                 k_proj_weight=None,              # type: Optional[Tensor]
                                 v_proj_weight=None,              # type: Optional[Tensor]
                                 static_k=None,                   # type: Optional[Tensor]
                                 static_v=None,                   # type: Optional[Tensor]
                                 memory_role_embedding=None       # type: Optional[Tensor]
                                 ):
    # type: (...) -> Tuple[Tensor, Optional[Tensor]]
    r"""
    Args:
        query, key, value: map a query and a set of key-value pairs to an output.
            See "Attention Is All You Need" for more details.
        embed_dim_to_check: total dimension of the model.
        num_heads: parallel attention heads.
        in_proj_weight, in_proj_bias: input projection weight and bias.
        bias_k, bias_v: bias of the key and value sequences to be added at dim=0.
        add_zero_attn: add a new batch of zeros to the key and
                       value sequences at dim=1.
        dropout_p: probability of an element to be zeroed.
        out_proj_weight, out_proj_bias: the output projection weight and bias.
        training: apply dropout if is ``True``.
        key_padding_mask: if provided, specified padding elements in the key will
            be ignored by the attention. This is an binary mask. When the value is True,
            the corresponding value on the attention layer will be filled with -inf.
        need_weights: output attn_output_weights.
        attn_mask: 2D or 3D mask that prevents attention to certain positions. A 2D mask will be broadcasted for all
            the batches while a 3D mask allows to specify a different mask for the entries of each batch.
        use_separate_proj_weight: the function accept the proj. weights for query, key,
            and value in different forms. If false, in_proj_weight will be used, which is
            a combination of q_proj_weight, k_proj_weight, v_proj_weight.
        q_proj_weight, k_proj_weight, v_proj_weight, in_proj_bias: input projection weight and bias.
        static_k, static_v: static key and value used for attention operators.


    Shape:
        Inputs:
        - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - key_padding_mask: :math:`(N, S)` where N is the batch size, S is the source sequence length.
          If a ByteTensor is provided, the non-zero positions will be ignored while the zero positions
          will be unchanged. If a BoolTensor is provided, the positions with the
          value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.
        - attn_mask: 2D mask :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
          3D mask :math:`(N*num_heads, L, S)` where N is the batch size, L is the target sequence length,
          S is the source sequence length. attn_mask ensures that position i is allowed to attend the unmasked
          positions. If a ByteTensor is provided, the non-zero positions are not allowed to attend
          while the zero positions will be unchanged. If a BoolTensor is provided, positions with ``True``
          are not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
          is provided, it will be added to the attention weight.
        - static_k: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.
        - static_v: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.

        Outputs:
        - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
          E is the embedding dimension.
        - attn_output_weights: :math:`(N, L, S)` where N is the batch size,
          L is the target sequence length, S is the source sequence length.
    """
    # if not torch.jit.is_scripting():
    #     tens_ops = (query, key, value, in_proj_weight, in_proj_bias, bias_k, bias_v,
    #                 out_proj_weight, out_proj_bias)
    #     if any([type(t) is not Tensor for t in tens_ops]) and has_torch_function(tens_ops):
    #         return handle_torch_function(
    #             multi_head_attention_forward, tens_ops, query, key, value,
    #             embed_dim_to_check, num_heads, in_proj_weight, in_proj_bias,
    #             bias_k, bias_v, add_zero_attn, dropout_p, out_proj_weight,
    #             out_proj_bias, training=training, key_padding_mask=key_padding_mask,
    #             need_weights=need_weights, attn_mask=attn_mask,
    #             use_separate_proj_weight=use_separate_proj_weight,
    #             q_proj_weight=q_proj_weight, k_proj_weight=k_proj_weight,
    #             v_proj_weight=v_proj_weight, static_k=static_k, static_v=static_v)
    tgt_len, bsz, embed_dim = query.size()
    assert embed_dim == embed_dim_to_check
    # allow MHA to have different sizes for the feature dimension
    assert key.size(0) == value.size(0) and key.size(1) == value.size(1)

    head_dim = embed_dim // num_heads
    assert head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
    scaling = float(head_dim) ** -0.5

    if not use_separate_proj_weight:
        if torch.equal(query, key) and torch.equal(key, value):
            # self-attention
            q, k, v = linear(query, in_proj_weight, in_proj_bias).chunk(3, dim=-1)

        elif torch.equal(key, value):
            # encoder-decoder attention
            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = 0
            _end = embed_dim
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            q = linear(query, _w, _b)

            if key is None:
                assert value is None
                k = None
                v = None
            else:

                # This is inline in_proj function with in_proj_weight and in_proj_bias
                _b = in_proj_bias
                _start = embed_dim
                _end = None
                _w = in_proj_weight[_start:, :]
                if _b is not None:
                    _b = _b[_start:]
                k, v = linear(key, _w, _b).chunk(2, dim=-1)

        else:
            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = 0
            _end = embed_dim
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            q = linear(query, _w, _b)

            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = embed_dim
            _end = embed_dim * 2
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            k = linear(key, _w, _b)

            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = embed_dim * 2
            _end = None
            _w = in_proj_weight[_start:, :]
            if _b is not None:
                _b = _b[_start:]
            v = linear(value, _w, _b)
    else:
        q_proj_weight_non_opt = torch.jit._unwrap_optional(q_proj_weight)
        len1, len2 = q_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == query.size(-1)

        k_proj_weight_non_opt = torch.jit._unwrap_optional(k_proj_weight)
        len1, len2 = k_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == key.size(-1)

        v_proj_weight_non_opt = torch.jit._unwrap_optional(v_proj_weight)
        len1, len2 = v_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == value.size(-1)

        if in_proj_bias is not None:
            q = linear(query, q_proj_weight_non_opt, in_proj_bias[0:embed_dim])
            k = linear(key, k_proj_weight_non_opt, in_proj_bias[embed_dim:(embed_dim * 2)])
            v = linear(value, v_proj_weight_non_opt, in_proj_bias[(embed_dim * 2):])
        else:
            q = linear(query, q_proj_weight_non_opt, in_proj_bias)
            k = linear(key, k_proj_weight_non_opt, in_proj_bias)
            v = linear(value, v_proj_weight_non_opt, in_proj_bias)
    q = q * scaling

    if attn_mask is not None:
        assert attn_mask.dtype == torch.float32 or attn_mask.dtype == torch.float64 or \
            attn_mask.dtype == torch.float16 or attn_mask.dtype == torch.uint8 or attn_mask.dtype == torch.bool, \
            'Only float, byte, and bool types are supported for attn_mask, not {}'.format(attn_mask.dtype)
        if attn_mask.dtype == torch.uint8:
            warnings.warn("Byte tensor for attn_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.")
            attn_mask = attn_mask.to(torch.bool)

        if attn_mask.dim() == 2:
            attn_mask = attn_mask.unsqueeze(0)
            if list(attn_mask.size()) != [1, query.size(0), key.size(0)]:
                raise RuntimeError('The size of the 2D attn_mask is not correct.')
        elif attn_mask.dim() == 3:
            if list(attn_mask.size()) != [bsz * num_heads, query.size(0), key.size(0)]:
                raise RuntimeError('The size of the 3D attn_mask is not correct.')
        else:
            raise RuntimeError("attn_mask's dimension {} is not supported".format(attn_mask.dim()))
        # attn_mask's dim is 3 now.

    # convert ByteTensor key_padding_mask to bool
    if key_padding_mask is not None and key_padding_mask.dtype == torch.uint8:
        warnings.warn("Byte tensor for key_padding_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.")
        key_padding_mask = key_padding_mask.to(torch.bool)

    if bias_k is not None and bias_v is not None:
        if static_k is None and static_v is None:
            k = torch.cat([k, bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = pad(attn_mask, (0, 1))
            if key_padding_mask is not None:
                key_padding_mask = pad(key_padding_mask, (0, 1))
        else:
            assert static_k is None, "bias cannot be added to static key."
            assert static_v is None, "bias cannot be added to static value."
    else:
        assert bias_k is None
        assert bias_v is None

    q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    if k is not None:
        k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    if v is not None:
        v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)

    if static_k is not None:
        assert static_k.size(0) == bsz * num_heads
        assert static_k.size(2) == head_dim
        k = static_k

    if static_v is not None:
        assert static_v.size(0) == bsz * num_heads
        assert static_v.size(2) == head_dim
        v = static_v

    src_len = k.size(1)

    if key_padding_mask is not None:
        assert key_padding_mask.size(0) == bsz
        assert key_padding_mask.size(1) == src_len

    if add_zero_attn:
        src_len += 1
        k = torch.cat([k, torch.zeros((k.size(0), 1) + k.size()[2:], dtype=k.dtype, device=k.device)], dim=1)
        v = torch.cat([v, torch.zeros((v.size(0), 1) + v.size()[2:], dtype=v.dtype, device=v.device)], dim=1)
        if attn_mask is not None:
            attn_mask = pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = pad(key_padding_mask, (0, 1))

    attn_output_weights = torch.bmm(q, k.transpose(1, 2))
    if memory_role_embedding is not None: # todo: check normalize & scaling
        memory_len = memory_role_embedding.shape[1]
        r = memory_role_embedding.view(tgt_len, memory_len, bsz * num_heads, head_dim) * scaling
        r = r.permute(2,0,1,3).contiguous() # (#heads, #query, #memory, #dim)
        attn_output_weights += torch.matmul(q.unsqueeze(2), r.transpose(3,2)).squeeze()
    assert list(attn_output_weights.size()) == [bsz * num_heads, tgt_len, src_len]

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_output_weights.masked_fill_(attn_mask, float('-inf'))
        else:
            attn_output_weights += attn_mask


    if key_padding_mask is not None:
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        attn_output_weights = attn_output_weights.masked_fill(
            key_padding_mask.unsqueeze(1).unsqueeze(2),
            float('-inf'),
        )
        attn_output_weights = attn_output_weights.view(bsz * num_heads, tgt_len, src_len)

    attn_output_weights = softmax(
        attn_output_weights, dim=-1)
    attn_output_weights = dropout(attn_output_weights, p=dropout_p, training=training)

    attn_output = torch.bmm(attn_output_weights, v)
    assert list(attn_output.size()) == [bsz * num_heads, tgt_len, head_dim]
    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
    attn_output = linear(attn_output, out_proj_weight, out_proj_bias)

    if need_weights:
        # average attention weights over heads
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        return attn_output, attn_output_weights.sum(dim=1) / num_heads
    else:
        return attn_output, None


class MultiheadAttention(torch.nn.Module):
    r"""Allows the model to jointly attend to information
    from different representation subspaces.
    See reference: Attention Is All You Need

    .. math::
        \text{MultiHead}(Q, K, V) = \text{Concat}(head_1,\dots,head_h)W^O
        \text{where} head_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)

    Args:
        embed_dim: total dimension of the model.
        num_heads: parallel attention heads.
        dropout: a Dropout layer on attn_output_weights. Default: 0.0.
        bias: add bias as module parameter. Default: True.
        add_bias_kv: add bias to the key and value sequences at dim=0.
        add_zero_attn: add a new batch of zeros to the key and
                       value sequences at dim=1.
        kdim: total number of features in key. Default: None.
        vdim: total number of features in value. Default: None.

        Note: if kdim and vdim are None, they will be set to embed_dim such that
        query, key, and value have the same number of features.

    Examples::

        >>> multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
        >>> attn_output, attn_output_weights = multihead_attn(query, key, value)
    """
    __annotations__ = {
        'bias_k': torch._jit_internal.Optional[torch.Tensor],
        'bias_v': torch._jit_internal.Optional[torch.Tensor],
    }

    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None):
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        if self._qkv_same_embed_dim is False:
            self.q_proj_weight = Parameter(torch.Tensor(embed_dim, embed_dim))
            self.k_proj_weight = Parameter(torch.Tensor(embed_dim, self.kdim))
            self.v_proj_weight = Parameter(torch.Tensor(embed_dim, self.vdim))
            self.register_parameter('in_proj_weight', None)
        else:
            self.in_proj_weight = Parameter(torch.empty(3 * embed_dim, embed_dim))
            self.register_parameter('q_proj_weight', None)
            self.register_parameter('k_proj_weight', None)
            self.register_parameter('v_proj_weight', None)

        if bias:
            self.in_proj_bias = Parameter(torch.empty(3 * embed_dim))
        else:
            self.register_parameter('in_proj_bias', None)
        self.out_proj = _LinearWithBias(embed_dim, embed_dim)

        if add_bias_kv:
            self.bias_k = Parameter(torch.empty(1, 1, embed_dim))
            self.bias_v = Parameter(torch.empty(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self._reset_parameters()

    def _reset_parameters(self):
        if self._qkv_same_embed_dim:
            xavier_uniform_(self.in_proj_weight)
        else:
            xavier_uniform_(self.q_proj_weight)
            xavier_uniform_(self.k_proj_weight)
            xavier_uniform_(self.v_proj_weight)

        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.)
            constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            xavier_normal_(self.bias_v)

    def __setstate__(self, state):
        # Support loading old MultiheadAttention checkpoints generated by v1.1.0
        if '_qkv_same_embed_dim' not in state:
            state['_qkv_same_embed_dim'] = True

        super(MultiheadAttention, self).__setstate__(state)

    def forward(self, query, key, value, key_padding_mask=None,
                need_weights=True, attn_mask=None, memory_role_embedding=None):
        # type: (Tensor, Tensor, Tensor, Optional[Tensor], bool, Optional[Tensor]) -> Tuple[Tensor, Optional[Tensor]]
        r"""
    Args:
        query, key, value: map a query and a set of key-value pairs to an output.
            See "Attention Is All You Need" for more details.
        key_padding_mask: if provided, specified padding elements in the key will
            be ignored by the attention. When given a binary mask and a value is True,
            the corresponding value on the attention layer will be ignored. When given
            a byte mask and a value is non-zero, the corresponding value on the attention
            layer will be ignored
        need_weights: output attn_output_weights.
        attn_mask: 2D or 3D mask that prevents attention to certain positions. A 2D mask will be broadcasted for all
            the batches while a 3D mask allows to specify a different mask for the entries of each batch.

    Shape:
        - Inputs:
        - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - key_padding_mask: :math:`(N, S)` where N is the batch size, S is the source sequence length.
          If a ByteTensor is provided, the non-zero positions will be ignored while the position
          with the zero positions will be unchanged. If a BoolTensor is provided, the positions with the
          value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.
        - attn_mask: 2D mask :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
          3D mask :math:`(N*num_heads, L, S)` where N is the batch size, L is the target sequence length,
          S is the source sequence length. attn_mask ensure that position i is allowed to attend the unmasked
          positions. If a ByteTensor is provided, the non-zero positions are not allowed to attend
          while the zero positions will be unchanged. If a BoolTensor is provided, positions with ``True``
          is not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
          is provided, it will be added to the attention weight.

        - Outputs:
        - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
          E is the embedding dimension.
        - attn_output_weights: :math:`(N, L, S)` where N is the batch size,
          L is the target sequence length, S is the source sequence length.
        """

        return multi_head_attention_forward_with_role(
            query, key, value, self.embed_dim, self.num_heads,
            self.in_proj_weight, self.in_proj_bias,
            self.bias_k, self.bias_v, self.add_zero_attn,
            self.dropout, self.out_proj.weight, self.out_proj.bias,
            training=self.training,
            key_padding_mask=key_padding_mask, need_weights=need_weights,
            attn_mask=attn_mask,
            memory_role_embedding=memory_role_embedding
        )

# 91
coco_obj_names = ['N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
# 80
hico_obj_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]
hico_obj_names = [coco_obj_names[id] for id in hico_obj_ids]
# 117
hico_action_names = ['adjust', 'assemble', 'block', 'blow', 'board', 'break', 'brush_with', 'buy', 'carry', 'catch', 'chase', 'check', 'clean', 'control', 'cook', 'cut', 'cut_with', 'direct', 'drag', 'dribble', 'drink_with', 'drive', 'dry', 'eat', 'eat_at', 'exit', 'feed', 'fill', 'flip', 'flush', 'fly', 'greet', 'grind', 'groom', 'herd', 'hit', 'hold', 'hop_on', 'hose', 'hug', 'hunt', 'inspect', 'install', 'jump', 'kick', 'kiss', 'lasso', 'launch', 'lick', 'lie_on', 'lift', 'light', 'load', 'lose', 'make', 'milk', 'move', 'no_interaction', 'open', 'operate', 'pack', 'paint', 'park', 'pay', 'peel', 'pet', 'pick', 'pick_up', 'point', 'pour', 'pull', 'push', 'race', 'read', 'release', 'repair', 'ride', 'row', 'run', 'sail', 'scratch', 'serve', 'set', 'shear', 'sign', 'sip', 'sit_at', 'sit_on', 'slide', 'smell', 'spin', 'squeeze', 'stab', 'stand_on', 'stand_under', 'stick', 'stir', 'stop_at', 'straddle', 'swing', 'tag', 'talk_on', 'teach', 'text_on', 'throw', 'tie', 'toast', 'train', 'turn', 'type_on', 'walk', 'wash', 'watch', 'wave', 'wear', 'wield', 'zip']

# 29 (25 valid)
vcoco_action_names = ['hold_obj', 'stand_agent', 'sit_instr', 'ride_instr', 'walk_agent', 'look_obj', 'hit_instr', 'hit_obj', 'eat_obj', 'eat_instr', 'jump_instr', 'lay_instr', 'talk_on_phone_instr', 'carry_obj', 'throw_obj', 'catch_obj', 'cut_instr', 'cut_obj', 'run_agent', 'work_on_computer_instr', 'ski_instr', 'surf_instr', 'skateboard_instr', 'smile_agent', 'drink_instr', 'kick_obj', 'point_instr', 'read_obj', 'snowboard_instr']
vcoco_valid_action_ids = [ 0,  2,  3,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 20, 21, 22, 24, 25, 26, 27, 28]

def check_annotation(samples, annotations, mode='train', rel_num=20, idx=0, dataset='hico'):
    obj_label_names = coco_obj_names
    if dataset == 'vcoco':
        action_label_names = vcoco_action_names
    else:
        action_label_names = hico_action_names

    img_tensors, img_masks = samples.decompose()
    h, w = (img_masks[idx].float() < 1).nonzero(as_tuple=False).max(0)[0].cpu() + 1

    img_tensor = img_tensors[idx,:,:h,:w].cpu().permute(1,2,0)
    img_tensor = (img_tensor - img_tensor.min()) / (img_tensor.max() - img_tensor.min())

    res = annotations[idx]
    org_h, org_w = res['orig_size'].cpu().float()
    boxes = res['boxes'].cpu()
    if mode == 'train':
        boxes = box_ops.box_cxcywh_to_xyxy(boxes)
        boxes = boxes * torch.tensor([w, h, w, h]).unsqueeze(0)
    else:
        boxes = boxes * torch.tensor([w/org_w, h/org_h, w/org_w, h/org_h]).unsqueeze(0)
    vg_obj_names = []
    for ind, x in enumerate(res['labels']):
        vg_obj_names.append(f"{obj_label_names[x]}({ind})")

    if 'hois' not in res:
        res['hois'] = res['relation_map'].nonzero(as_tuple=False)
    rel_pairs = res['hois'][:rel_num, :2].cpu()
    rel_labels = res['hois'][:rel_num, 2].cpu()

    # list relations
    rel_strs = ''
    for i, rel in enumerate(rel_pairs): # print relation triplets
        if dataset == 'vcoco' and rel_labels[i] not in vcoco_valid_action_ids: continue
        rel_strs += (f"{vg_obj_names[rel[0]]} ---{action_label_names[rel_labels[i]]}----> {vg_obj_names[rel[1]]}\n")

    # draw images
    plt.imshow(img_tensor)
    for ind, bbox in enumerate(boxes):
        x1, y1, x2, y2 = bbox
        rect = patches.Rectangle((x1,y1), x2-x1+1, y2-y1+1, linewidth=1, edgecolor='r', facecolor='none')
        plt.gca().add_patch(rect)
        txt = plt.text(x1-10, y1-10, vg_obj_names[ind], color='black')
        txt.set_path_effects([PathEffects.withStroke(linewidth=5, foreground='w')])
    plt.gca().yaxis.set_label_position("right")
    # plt.ylabel(rel_strs, rotation=0, labelpad=140, fontsize=8, loc='top')
    plt.ylabel(rel_strs, rotation=0, labelpad=140, fontsize=8)
    plt.show()


COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]
def plot_cross_attention(samples, results, targets, attn_maps, idx=0, dataset='hico', topk_qids=None):
    if dataset == 'vcoco':
        action_label_names = vcoco_action_names
    else:
        action_label_names = hico_action_names

    pil_imgs, masks = samples.decompose()
    pil_img, mask, attn_map = pil_imgs[idx], masks[idx], attn_maps[idx]

    pil_img = ((pil_img - pil_img.min()) / (pil_img.max() - pil_img.min())).permute(1,2,0).cpu().numpy()
    h, w = (~mask).float().nonzero(as_tuple=False).max(0)[0] + 1
    pil_img = pil_img[:h, :w]

    boxes = box_ops.box_cxcywh_to_xyxy(results['pred_boxes'][idx].cpu()) * torch.tensor([w,h,w,h])
    box_scores, box_labels = results['pred_logits'][idx].softmax(-1)[:, :-1].cpu().max(-1)

    pair_id_counts = 4
    if topk_qids is not None:
        plot_qids = []
        for q in topk_qids:
            if q not in plot_qids and len(plot_qids) < pair_id_counts :
                plot_qids.append(q)
    else:
        plot_qids = list(range(pair_id_counts))
    plot_num = pair_id_counts+1
    _, axes = plt.subplots(1, plot_num, figsize=(9*plot_num, 7))
    ######## plt boxes ##########
    # axes[0].set_title(f"image_id={targets[idx]['image_id'].item()}")
    axes[0].imshow(pil_img)
    # colors = COLORS * 100
    # for sc, l, (xmin, ymin, xmax, ymax), c in zip(box_scores, box_labels, boxes, colors):
    #     if sc < 0.9: continue
    #     axes[0].add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, color=c, linewidth=3))
    #     text = f'{coco_obj_names[l]}({sc:0.2f})'
    #     axes[0].text(xmin, ymin, text, fontsize=14, bbox=dict(facecolor='yellow', alpha=0.5))
    axes[0].set_axis_off()

    plot_qids.sort()
    for aidx, pair_id in enumerate(plot_qids):
        ax = axes[aidx+1]
        ######## specific relation ##########
        subj_id, obj_id = results['pred_rel_pairs'][idx][pair_id]
        action_probs = results['pred_actions'][idx][pair_id].sigmoid().cpu()
        if dataset == 'vcoco':
            for k in range(len(action_probs)):
                if k not in vcoco_valid_action_ids: action_probs[k] = -1
        action_scores, action_labels = action_probs.sort(descending=True)
        # pair_action_label_names = '\n'.join([f"{action_label_names[action_labels[k]]} ({action_scores[k]: 0.2f})" for k in range(len(action_labels)) if k<3 or action_scores[k] > 0.5])
        pair_action_label_names = f"qid={pair_id}:   " + ', '.join([f"{action_label_names[action_labels[k]]}({action_scores[k]: .2f})" for k in range(len(action_labels)) if k<3])

        ######## img with cross attention ##########
        featmap_scale = 32
        fH, fW = int(np.ceil(masks.shape[1] / featmap_scale)), int(np.ceil(masks.shape[2] / featmap_scale))
        fh, fw = h // featmap_scale + 1, w // featmap_scale + 1

        pair_attention = attn_map[0, pair_id].view(fH, fW)[:fh, :fw]
        pair_attention = (pair_attention - pair_attention.min()) / (pair_attention.max() - pair_attention.min())
        pair_attention = F.interpolate(pair_attention.unsqueeze(0).unsqueeze(1), size=pil_img.shape[:-1], mode='bilinear').squeeze().cpu().numpy()

        # show_img = pil_img * 0.3 + 0.7 * pair_attention[:,:,None]
        ax.imshow(pair_attention, vmax=0.7)
        ax.imshow(pil_img, alpha=0.15)

        ######## relation pair boxes ##########
        # head
        hxmin, hymin, hxmax, hymax = boxes[subj_id]
        ax.add_patch(plt.Rectangle((hxmin, hymin), hxmax - hxmin, hymax - hymin, fill=False, color='red', linewidth=3))
        # text = f'{coco_obj_names[box_labels[subj_id]]}_{subj_id}({box_scores[subj_id]: 0.2f})'
        text = f'{coco_obj_names[box_labels[subj_id]]}_{subj_id}'
        ax.text(hxmin, hymin, text, fontsize=22, bbox=dict(facecolor='red', alpha=0.3))

        # tail
        txmin, tymin, txmax, tymax = boxes[obj_id]
        ax.add_patch(plt.Rectangle((txmin, tymin), txmax - txmin, tymax - tymin, fill=False, color='yellow', linewidth=3))
        # text = f'{coco_obj_names[box_labels[obj_id]]}_{obj_id}({box_scores[obj_id]: 0.2f})'
        text = f'{coco_obj_names[box_labels[obj_id]]}_{obj_id}'
        ax.text(txmin, tymin, text, fontsize=22, bbox=dict(facecolor='yellow', alpha=0.3))

        # head -> tail arrow
        ax1, ay1, ax2, ay2 = (hxmin+hxmax) / 2, (hymin+hymax) / 2, (txmin+txmax) / 2, (tymin+tymax) / 2
        # axes[1].add_patch(patches.FancyArrow(ax1, ay1, ax2-ax1, ay2-ay1, color='blue', linewidth=5))
        # axes[1].arrow(ax1, ay1, ax2-ax1, ay2-ay1, color='blue')
        ax.plot(ax1, ay1, 'o', color='red', markersize=20)
        ax.arrow(ax1, ay1, ax2-ax1, ay2-ay1, head_width=10, head_length=10, color='orange', linewidth=8)

        ax.set_title(pair_action_label_names, fontsize=16)
        ax.set_axis_off()
        print(pair_id)

    plt.show()
    print(pair_action_label_names)

def plot_hoi_results(samples, results, targets, args=None, idx=0):
    pil_imgs, masks = samples.decompose()
    pil_img, mask= pil_imgs[idx], masks[idx]

    pil_img = ((pil_img - pil_img.min()) / (pil_img.max() - pil_img.min())).permute(1,2,0).cpu().numpy()
    h, w = (~mask).float().nonzero(as_tuple=False).max(0)[0] + 1
    pil_img = pil_img[:h, :w]

    boxes = box_ops.box_cxcywh_to_xyxy(results['pred_boxes'][idx].cpu()) * torch.tensor([w,h,w,h])
    box_scores, box_labels = results['pred_logits'][idx].softmax(-1)[:, :-1].cpu().max(-1)

    ####### proposals ##########
    props = results['pred_rel_pairs'][idx]
    if 'pred_action_exists' in results:
        prop_scores = results['pred_action_exists'][idx].sigmoid()
    prop_obj_ids = props.unique()
    prop_strs = '====================================== Proposals ======================================\n'
    q_name_list = []
    for qid, (head_id, tail_id) in enumerate(props):
        if 'pred_action_exists' in results: # with proposal score
            prop_strs += f"q={qid} ({prop_scores[qid]: .2f}):\t{coco_obj_names[box_labels[head_id]]}_{head_id} ({box_scores[head_id]: .2f})\t ======>\t\t{coco_obj_names[box_labels[tail_id]]}_{tail_id}({box_scores[tail_id]: .2f})\n"
        else:
            prop_strs += f"q={qid}:\t{coco_obj_names[box_labels[head_id]]}_{head_id} ({box_scores[head_id]: .2f})\t ======>\t\t{coco_obj_names[box_labels[tail_id]]}_{tail_id}({box_scores[tail_id]: .2f})\n"
        q_name_list.append(f"{coco_obj_names[box_labels[head_id]]}_{head_id}->{coco_obj_names[box_labels[tail_id]]}_{tail_id}")
    print(prop_strs)

    ####### top predicted hois ##########
    K = 100
    tail_scores = box_scores[results['pred_rel_pairs'][idx][:, -1]]
    verb_scores = results['pred_actions'][idx].sigmoid().cpu()
    hoi_scores = verb_scores * tail_scores.unsqueeze(1)

    # apply mask
    tail_labels = box_labels[results['pred_rel_pairs'][idx][:, -1]]
    tail_labels = [hico_obj_ids.index(x) for x in tail_labels] # to hico obj labels
    mask = args.correct_mat.transpose(1,0).cpu()[tail_labels]
    hoi_scores *= mask

    qnum, action_num = hoi_scores.shape
    _, sort_ids = hoi_scores.view(-1).sort(descending=True)
    topk_qids, topk_actions = sort_ids[:K] // action_num, sort_ids[:K] % action_num

    rel_strs = '====================================== Relations ======================================\n'
    for qid, qaction in zip(topk_qids, topk_actions):
        head_id, tail_id = results['pred_rel_pairs'][idx][qid]
        rel_strs += f"q={qid}:\t{coco_obj_names[box_labels[head_id]]}_{head_id} ({box_scores[head_id]: .2f})\t === {hico_action_names[qaction]} ({verb_scores[qid, qaction]: .2f}) ===>\t\t{coco_obj_names[box_labels[tail_id]]}_{tail_id}({box_scores[tail_id]: .2f})\n"
    print(rel_strs)

    ######## detr match ##########
    if 'det2gt_indices' in results and results['det2gt_indices'] is not None:
        print('GT -> DET match:')
        gt2det_dict = {int(g): int(d) for d, g in zip(*results['det2gt_indices'][idx])}
        for i in range(len(gt2det_dict)): print(f"{i} ---> {gt2det_dict[i]}")

    ######## plt detected boxes ##########
    plt.imshow(pil_img)
    colors = COLORS * 100
    for id, (sc, l, (xmin, ymin, xmax, ymax), c) in enumerate(zip(box_scores, box_labels, boxes, colors)):
        # if id in prop_obj_ids or sc > 0.5:
        if sc > 0.5:
            plt.gca().add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, color=c))
            # text = f'{coco_obj_names[l]}_{str(id)}({sc:0.2f})'
            text = f'{coco_obj_names[l]}_{str(id)}'
            plt.text(xmin, ymin, text, fontsize=10, bbox=dict(facecolor='white', alpha=0.5))
    plt.axis('off')

    # plt.gca().yaxis.set_label_position("right")
    # plt.ylabel(rel_strs, rotation=0, labelpad=140, fontsize=8, loc='top')\
    plt.title(f"image_id={targets[idx]['id']}")
    plt.show()
    print('plot_hoi_results')
    return topk_qids.tolist(), q_name_list
