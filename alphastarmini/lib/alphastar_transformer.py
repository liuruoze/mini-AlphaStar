#!/usr/bin/env python
# -*- coding: utf-8 -*-

" The transformer used for mini-alphastar."

import torch
import torch.nn as nn
import torch.nn.functional as F

# Deprecated
# from alphastarmini.third.transformer.SubLayers import MultiHeadAttention, PositionwiseFeedForward

from alphastarmini.lib.transformer_layer import MultiHeadAttention
from alphastarmini.lib.transformer_layer import PositionwiseFeedForward

__author__ = "Ruo-Ze Liu"

debug = True


class Transformer(nn.Module):
    ''' AlphaStar transformer composed with only three encoder layers '''
    # refactored by reference to https://github.com/metataro/sc2_imitation_learning

    # default parameter from AlphaStar
    def __init__(
            self, d_model=256, d_inner=1024,
            n_layers=3, n_head=2, d_k=128, d_v=128, dropout=0.1):

        super().__init__()

        self.encoder = Encoder(
            d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            dropout=dropout)

        # for p in self.parameters():
        #     if p.dim() > 1:
        #         nn.init.xavier_uniform_(p)

    def forward(self, x, mask=None):
        enc_output, *_ = self.encoder(x, mask=mask)

        return enc_output


class Encoder(nn.Module):
    ''' A alphastar encoder model with self attention mechanism. '''

    # default parameter from AlphaStar
    def __init__(
            self, n_layers=3, n_head=2, d_k=128, d_v=128,
            d_model=256, d_inner=1024, dropout=0.1):

        super().__init__()

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

        # note "unbiased=False" will affect the results
        # layer_norm is b = (a - torch.mean(a))/(torch.var(a, unbiased=False)**0.5) * 1.0 + 0.0

    def forward(self, x, mask=None):
        # -- Forward

        enc_output = x
        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output, mask=mask)

        del enc_slf_attn

        return enc_output,


class EncoderLayer(nn.Module):
    '''     
    '''

    # default parameter from AlphaStar
    def __init__(self, d_model=256, d_inner=1024, n_head=2, d_k=128, d_v=128, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)

        self.ln1 = nn.LayerNorm(d_model, eps=1e-6)
        self.ln2 = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, x, mask=None):
        att_out, enc_slf_attn = self.slf_attn(x, x, x, mask=mask)

        att_out = self.drop1(att_out)
        out_1 = self.ln1(x + att_out)

        ffn_out = self.pos_ffn(out_1)

        ffn_out = self.drop2(ffn_out)
        out = self.ln2(out_1 + ffn_out)

        del att_out, out_1, ffn_out

        return out, enc_slf_attn


def test():
    pass


if __name__ == '__main__':
    test()
