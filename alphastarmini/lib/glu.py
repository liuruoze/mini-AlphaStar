#!/usr/bin/env python
# -*- coding: utf-8 -*-

" Helper Modules: Gating Linear Unit (GLU) "

import torch
import torch.nn as nn
import torch.nn.functional as F

__author__ = "Ruo-Ze Liu"

debug = False


class GLU(nn.Module):
    '''
    Gating Linear Unit.
    Inputs: input, context, output_size
    '''

    def __init__(self, input_size=384, context_size=1024,
                 output_size=1024):
        super().__init__()
        self.fc_1 = nn.Linear(context_size, input_size)
        self.fc_2 = nn.Linear(input_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, context):
        # context shape: [batch_size x context_size]
        gate = self.sigmoid(self.fc_1(context))
        # gate shape: [batch_size x input_size]

        # The line is the same as below: gated_input = torch.mul(gate, x)
        # x shape: [batch_size x input_size]
        gated_input = gate * x

        # gated_input shape: [batch_size x input_size]
        output = self.fc_2(gated_input)

        del context, x, gate, gated_input

        return output


def test():
    context = torch.randn(5, 32)
    x = torch.randn(5, 16)
    output_size = 24

    GLU = GatingLinearUnit(input_size=x.shape[-1], context_size=context.shape[-1],
                           output_size=output_size)
    print('context:', context) if debug else None
    print('x:', x) if debug else None

    output = GLU.forward(x, context)
    print('output:', output) if debug else None

    print("This is a test!") if debug else None


if __name__ == '__main__':
    test()
