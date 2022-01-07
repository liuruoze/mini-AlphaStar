''' Define the sublayers in encoder/decoder layer '''
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

__original__author__ = "Yu-Hsiang Huang"

# modified refernect mostly to https://github.com/opendilab/DI-star in distar.model.alphastar.module_utils.Transformer
__modified__ = "Ruo-Ze Liu"


debug = True


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1, bias_value=-1e9):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.biasval = bias_value

    def forward(self, q, k, v, mask=None):

        # q: (b, n, lq, dk)
        # k: (b, n, lk, dk)
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        # atten: (b, n, lq, lk),
        if mask is not None:
            attn = attn.masked_fill(mask == 0, self.biasval)
            del mask

        attn = self.dropout(F.softmax(attn, dim=-1))

        # v: (b, n, lv, dv)
        # r: (b, n, lq, dv)
        r = torch.matmul(attn, v)

        return r, attn


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        # pre-attention projection
        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=True)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=True)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=True)

        # after-attention projection
        self.fc = nn.Linear(n_head * d_v, d_model, bias=True)

        # attention
        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

    def forward(self, q, k, v, mask=None):
        # q: (b, lq, dm)
        # k: (b, lk, dm)
        # v: (b, lv, dm)

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        size_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        # pass through the pre-attention projection
        # separate different heads

        # after that q: (b, lq, n, dk)
        q = self.w_qs(q).view(size_b, len_q, n_head, d_k)

        k = self.w_ks(k).view(size_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(size_b, len_v, n_head, d_v)

        # transpose for attention dot product: (b, n, lq, dk)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        q, attn = self.attention(q, k, v, mask=mask)

        # q: (b, n, lq, dk), k: (b, n, lk, dk), atten = q \matmul k^t = (b, n, lq, lk),
        # v: (b, n, lv, dv), assert lk = lv
        # atten \matmul v = (b, n, lq, dv)

        # transpose to move the head dimension back: (b, lq, n, dv)
        # combine the last two dimensions to concatenate all the heads together: (b, lq, (n*dv))
        q = q.transpose(1, 2).contiguous().view(size_b, len_q, -1)

        # q: (b, lq, (n*dv)) \matmul ((n*dv), dm) = (b, lq, dm)
        # note, q has the same shape as when it enter in
        q = self.fc(q)

        del mask, k, v, 

        return q, attn


class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)  # position-wise
        self.w_2 = nn.Linear(d_hid, d_in)  # position-wise

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)

        return x
