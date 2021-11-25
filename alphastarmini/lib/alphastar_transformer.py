#!/usr/bin/env python
# -*- coding: utf-8 -*-

" The transformer used for mini-alphastar."

import torch
import torch.nn as nn
import torch.nn.functional as F

from alphastarmini.third.transformer.SubLayers import MultiHeadAttention, PositionwiseFeedForward


__author__ = "Ruo-Ze Liu"

debug = True


class Transformer(nn.Module):
    ''' AlphaStar transformer composed with only three encoder layers '''

    # default parameter from AlphaStar
    def __init__(
            self, d_model=256, d_inner=1024,
            n_layers=3, n_head=2, d_k=128, d_v=128, dropout=0.1):

        super().__init__()

        self.encoder = Encoder(
            d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            dropout=dropout)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        enc_output, *_ = self.encoder(x)

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

        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

        # note "unbiased=False" will affect the results
        # layer_norm is b = (a - torch.mean(a))/(torch.var(a, unbiased=False)**0.5) * 1.0 + 0.0

    def forward(self, x):
        # -- Forward

        enc_output = x
        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output)

        enc_output = self.layer_norm(enc_output)

        return enc_output,


class EncoderLayer(nn.Module):
    '''     
    '''

    # default parameter from AlphaStar
    def __init__(self, d_model=256, d_inner=1024, n_head=2, d_k=128, d_v=128, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask)
        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn


def test():
    pass


if __name__ == '__main__':
    test()
