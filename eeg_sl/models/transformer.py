"""Transformer building blocks used by EEG2Source2EEG_autoregressive.

These are the standard "Annotated Transformer" style modules (encoder/decoder
stacks, multi-head attention, position-wise FFN, positional encoding) plus a few
projector heads.  Variants that were never trained are omitted.
"""

import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.functional import scaled_dot_product_attention
from torch.nn.attention import SDPBackend, sdpa_kernel


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


# --------------------------------------------------------------------------- #
#  Norm / sublayer
# --------------------------------------------------------------------------- #
class LayerNorm(nn.Module):
    "Construct a layernorm module."
    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


# --------------------------------------------------------------------------- #
#  Encoder
# --------------------------------------------------------------------------- #
class Encoder(nn.Module):
    "Core encoder is a stack of N layers."
    def __init__(self, layer, N):
        super().__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        atten_list = []
        for layer in self.layers:
            x, atten = layer(x, mask)
            atten_list.append(atten)
        return self.norm(x), atten_list


class EncoderLayer(nn.Module):
    "Encoder layer: self-attn + feed forward (pre-norm residual)."
    def __init__(self, size, self_attn, feed_forward, dropout):
        super().__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.size = size

    def forward(self, x, mask):
        hidden = self.norm(x)
        hidden, atten = self.self_attn(hidden, hidden, hidden, mask)
        hidden = self.dropout(hidden)
        x = x + hidden
        hidden = self.norm(x)
        hidden = self.feed_forward(hidden)
        hidden = self.dropout(hidden)
        x = x + hidden
        return x, atten


# --------------------------------------------------------------------------- #
#  Decoder
# --------------------------------------------------------------------------- #
class Decoder(nn.Module):
    "Generic N layer decoder with masking."
    def __init__(self, layer, N):
        super().__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        atten_list = []
        for layer in self.layers:
            x, atten = layer(x, memory, src_mask, tgt_mask)
            atten_list.append(atten)
        return self.norm(x), atten_list


class DecoderLayer(nn.Module):
    "Decoder layer: self-attn, cross-attn, feed forward (pre-norm residual)."
    def __init__(self, size, self_attn, cross_attn, feed_forward, dropout):
        super().__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)
        self.size = size
        self.self_attn = self_attn
        self.cross_attn = cross_attn
        self.feed_forward = feed_forward

    def forward(self, x, memory, src_mask, tgt_mask):
        # self attention
        hidden = self.norm(x)
        hidden, atten_1 = self.self_attn(hidden, hidden, hidden, tgt_mask)
        hidden = self.dropout(hidden)
        x = x + hidden
        # cross attention to encoder memory
        hidden = self.norm(x)
        hidden, atten_2 = self.cross_attn(hidden, memory, memory, src_mask)
        hidden = self.dropout(hidden)
        x = x + hidden
        # feed forward
        hidden = self.norm(x)
        hidden = self.feed_forward(hidden)
        hidden = self.dropout(hidden)
        x = x + hidden
        return x, (atten_1, atten_2)


# --------------------------------------------------------------------------- #
#  Attention
# --------------------------------------------------------------------------- #
def attention(query, key, value, mask=None, dropout=None):
    "Scaled Dot Product Attention."
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    "Vanilla multi-head attention (the default; sdp_attention=False)."
    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        query, key, value = [
            l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for l, x in zip(self.linears, (query, key, value))
        ]
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x), self.attn


class MultiHeaded_FlashAttention(nn.Module):
    "SDPA / flash-attention variant (used when config.sdp_attention=True)."
    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        assert d_model % h == 0
        self.h = h
        self.d_k = d_model // h
        self.q_proj = nn.Linear(d_model, d_model, bias=True)
        self.k_proj = nn.Linear(d_model, d_model, bias=True)
        self.v_proj = nn.Linear(d_model, d_model, bias=True)
        self.o_proj = nn.Linear(d_model, d_model, bias=True)
        self.dropout = dropout

    def _shape(self, x):
        B, L, D = x.shape
        return x.view(B, L, self.h, self.d_k).permute(0, 2, 1, 3)

    def forward(self, query, key, value, attn_mask=None):
        q = self._shape(self.q_proj(query))
        k = self._shape(self.k_proj(key))
        v = self._shape(self.v_proj(value))
        B, h, Lq, d = q.shape
        with sdpa_kernel([SDPBackend.FLASH_ATTENTION,
                          SDPBackend.EFFICIENT_ATTENTION,
                          SDPBackend.MATH]):
            y = scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=False,
            )
        y = y.permute(0, 2, 1, 3).contiguous().view(B, Lq, h * d)
        return self.o_proj(y), None


# --------------------------------------------------------------------------- #
#  Feed-forward / projectors
# --------------------------------------------------------------------------- #
class PositionwiseFeedForward(nn.Module):
    "FFN: Linear -> ReLU -> Dropout -> Linear."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class MLP_projector(nn.Module):
    "3-layer MLP projector (in_dim -> hidden -> hidden -> out_dim)."
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.layer1 = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(inplace=True))
        self.layer2 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True))
        self.layer3 = nn.Sequential(nn.Linear(hidden_dim, out_dim))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


# ---- optional generator heads (used when config.use_generator != "linear") ---
class GEGLU(nn.Module):
    # https://arxiv.org/abs/2002.05202
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return x + self.fn(x)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x):
        return self.fn(self.norm(x))


class MLPProjectorPlus(nn.Module):
    "Gated-MLP projector (PreNorm + GEGLU + residual), d_model -> time."
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.1, film_dim=None):
        super().__init__()
        self.film = None
        if film_dim is not None:
            self.film = nn.Sequential(nn.Linear(film_dim, in_dim * 2), nn.Tanh())
        self.block1 = Residual(PreNorm(in_dim, nn.Sequential(
            GEGLU(in_dim, hidden_dim), nn.Dropout(dropout), nn.Linear(hidden_dim, in_dim))))
        self.block2 = Residual(PreNorm(in_dim, nn.Sequential(
            GEGLU(in_dim, hidden_dim), nn.Dropout(dropout), nn.Linear(hidden_dim, in_dim))))
        self.out = nn.Sequential(
            nn.LayerNorm(in_dim), nn.Linear(in_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, out_dim))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / math.sqrt(fan_in)
                    nn.init.uniform_(m.bias, -bound, bound)

    def forward(self, x, film_ctx=None):
        if self.film is not None and film_ctx is not None:
            gamma, beta = self.film(film_ctx).chunk(2, dim=-1)
            while gamma.dim() < x.dim():
                gamma = gamma.unsqueeze(1)
                beta = beta.unsqueeze(1)
            x = x * (1 + gamma) + beta
        x = self.block1(x)
        x = self.block2(x)
        return self.out(x)


class BasisProjector(nn.Module):
    "Reconstruct a time series from a fixed (DCT/sine) basis."
    def __init__(self, in_dim, hidden_dim, out_len, K=128, basis_type="dct"):
        super().__init__()
        self.out_len = out_len
        self.K = K
        self.coeff_head = nn.Sequential(
            nn.LayerNorm(in_dim), nn.Linear(in_dim, hidden_dim),
            nn.GELU(), nn.Linear(hidden_dim, K))
        B = self._build_basis(out_len, K, basis_type=basis_type)
        self.register_buffer("basis", B)

    @staticmethod
    def _build_basis(T, K, basis_type="dct"):
        t = torch.arange(T).float().unsqueeze(1)
        if basis_type == "sine":
            freqs = torch.arange(1, K // 2 + 1).float().unsqueeze(0)
            s = torch.sin(2 * math.pi * freqs * t / T)
            c = torch.cos(2 * math.pi * freqs * t / T)
            B = torch.cat([s, c], dim=1)
            if B.shape[1] < K:
                B = F.pad(B, (0, K - B.shape[1]))
        else:
            k = torch.arange(K).float().unsqueeze(0)
            B = torch.cos(math.pi * (t + 0.5) * k / T)
        return F.normalize(B, dim=0)

    def forward(self, x):
        coeffs = self.coeff_head(x)
        return coeffs @ self.basis.t()


class HybridProjector(nn.Module):
    "Learnable mix of an MLP projector and a fixed-basis projector."
    def __init__(self, in_dim, hidden_dim, out_len, K=128):
        super().__init__()
        self.mlp = MLPProjectorPlus(in_dim, hidden_dim, out_len)
        self.basis = BasisProjector(in_dim, hidden_dim, out_len, K=K, basis_type="dct")
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, x, film_ctx=None):
        y1 = self.mlp(x, film_ctx)
        y2 = self.basis(x)
        a = torch.sigmoid(self.alpha)
        return a * y1 + (1 - a) * y2


# --------------------------------------------------------------------------- #
#  Positional encoding
# --------------------------------------------------------------------------- #
class PositionalEncoding(nn.Module):
    "Sinusoidal positional encoding (used when pos_embedding_type='sinusoid')."
    def __init__(self, d_model, dropout, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x, dim=3):
        if dim == 3:
            x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        elif dim == 4:
            x = x + Variable(self.pe[:, :x.size(2)], requires_grad=False)
        return self.dropout(x)
