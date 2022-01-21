#!/usr/bin/env python
# -*- coding: utf-8 -*-

" Action Type Head."

import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from pysc2.lib.actions import RAW_FUNCTIONS
from alphastarmini.core.arch.spatial_encoder import ResBlock1D
from alphastarmini.lib.glu import GLU

from alphastarmini.lib import utils as L

from alphastarmini.lib.hyper_parameters import Arch_Hyper_Parameters as AHP
from alphastarmini.lib.hyper_parameters import Label_Size as LS

from alphastarmini.lib.sc2 import raw_actions_mapping_protoss as RAMP

__author__ = "Ruo-Ze Liu"

debug = False


class ActionTypeHead(nn.Module):
    '''
    Inputs: lstm_output, scalar_context
    Outputs:
        action_type_logits - The logits corresponding to the probabilities of taking each action
        action_type - The action_type sampled from the action_type_logits
        autoregressive_embedding - Embedding that combines information from `lstm_output` and all previous sampled arguments. 
        To see the order arguments are sampled in, refer to the network diagram
    '''

    def __init__(self, lstm_dim=AHP.lstm_hidden_dim, n_resblocks=AHP.n_resblocks, 
                 is_sl_training=True, temperature=AHP.temperature, original_256=AHP.original_256,
                 max_action_num=LS.action_type_encoding, context_size=AHP.context_size, 
                 autoregressive_embedding_size=AHP.autoregressive_embedding_size,
                 use_action_type_mask=AHP.use_action_type_mask):
        super().__init__()
        self.is_sl_training = is_sl_training
        self.temperature = temperature

        self.embed_fc = nn.Linear(lstm_dim, original_256)  # with relu
        self.resblock_stack = nn.ModuleList([
            ResBlock1D(inplanes=original_256, planes=original_256, seq_len=1)
            for _ in range(n_resblocks)])

        self.max_action_num = max_action_num
        self.glu_1 = GLU(input_size=original_256, context_size=context_size,
                         output_size=max_action_num)

        self.fc_1 = nn.Linear(max_action_num, original_256)
        self.glu_2 = GLU(input_size=original_256, context_size=context_size,
                         output_size=autoregressive_embedding_size)
        self.glu_3 = GLU(input_size=lstm_dim, context_size=context_size,
                         output_size=autoregressive_embedding_size)
        self.softmax = nn.Softmax(dim=-1)

        self.use_action_type_mask = use_action_type_mask
        self.is_rl_training = False

    def set_rl_training(self, staus):
        self.is_rl_training = staus

    def forward(self, lstm_output, scalar_context, action_type_mask=None, action_type=None):
        batch_size = lstm_output.shape[0]

        # AlphaStar: The action type head embeds `lstm_output` into a 1D tensor of size 256
        x = self.embed_fc(lstm_output)

        # AlphaStar: passes it through 16 ResBlocks with layer normalization each of size 256, and applies a ReLU. 
        # QUESTION: There is no map, how to use resblocks?
        # ANSWER: USE resblock1D
        # input shape is [batch_size x seq_size x embedding_size]
        # note that embedding_size is equal to channel_size in conv1d
        # we change this to [batch_size x embedding_size x seq_size]
        x = x.unsqueeze(-1)
        for resblock in self.resblock_stack:
            x = resblock(x)
        x = F.relu(x)
        x = x.squeeze(-1)

        # AlphaStar: The output is converted to a tensor with one logit for each possible 
        # action type through a `GLU` gated by `scalar_context`.
        action_type_logits = self.glu_1(x, scalar_context)

        # inspired by the DI-star project, in action_type_head
        if self.is_rl_training and self.use_action_type_mask and action_type_mask is not None:
            action_type_mask = action_type_mask.bool()
            if action_type is not None:       
                for i, a in enumerate(action_type):
                    action_type_mask[i, a.item()] = True
            action_type_logits = action_type_logits + (~action_type_mask * (-1e9))
            del action_type_mask

        print('action_type_logits:', action_type_logits) if debug else None
        print('action_type_logits.shape:', action_type_logits.shape) if debug else None

        # AlphaStar: `action_type` is sampled from these logits using a multinomial with temperature 0.8. 
        # Note that during supervised learning, `action_type` will be the ground truth human action 
        # type, and temperature is 1.0 (and similarly for all other arguments).
        temperature = self.temperature if self.is_rl_training else 1
        action_type_logits = action_type_logits / temperature
        print('action_type_logits:', action_type_logits) if debug else None
        print('action_type_logits.shape:', action_type_logits.shape) if debug else None

        # note, torch.multinomial need samples to non-negative, finite and have a non-zero sum
        # which is different with tf.multinomial which can accept negative values like log(action_type_probs)
        if action_type is None:
            action_type_probs = self.softmax(action_type_logits)
            action_type_probs = action_type_probs.reshape(batch_size, -1)
            print('action_type_probs:', action_type_probs) if debug else None
            print('action_type_probs.shape:', action_type_probs.shape) if debug else None

            action_type = torch.multinomial(action_type_probs, 1)
            action_type = action_type.reshape(batch_size, -1)
            del action_type_probs

        # change action_type to one_hot version
        action_type_one_hot = L.tensor_one_hot(action_type, self.max_action_num)
        action_type_one_hot = action_type_one_hot.squeeze(-2)

        # AlphaStar: `autoregressive_embedding` is then generated by first applying a ReLU 
        # and linear layer of size 256 to the one-hot version of `action_type`
        z = F.relu(self.fc_1(action_type_one_hot))

        # AlphaStar: and projecting it to a 1D tensor of size 1024 through a `GLU` gated by `scalar_context`.
        z = self.glu_2(z, scalar_context)

        # AlphaStar: That projection is added to another projection of `lstm_output` into a 1D tensor of size 
        # 1024 gated by `scalar_context` to yield `autoregressive_embedding`.
        t = self.glu_3(lstm_output, scalar_context)

        # the add operation may auto broadcasting, so we need an assert test
        autoregressive_embedding = z + t

        del action_type_one_hot, lstm_output, scalar_context, x, z, t

        return action_type_logits, action_type, autoregressive_embedding


def test():
    batch_size = 2
    lstm_output = torch.randn(batch_size * AHP.sequence_length, AHP.lstm_hidden_dim)
    scalar_context = torch.randn(batch_size * AHP.sequence_length, AHP.context_size)
    action_type_head = ActionTypeHead()

    print("lstm_output:", lstm_output) if debug else None
    print("lstm_output.shape:", lstm_output.shape) if debug else None

    print("scalar_context:", scalar_context) if debug else None
    print("scalar_context.shape:", scalar_context.shape) if debug else None

    action_type_logits, action_type, autoregressive_embedding = action_type_head.forward(lstm_output, scalar_context)

    print("action_type_logits:", action_type_logits) if debug else None
    print("action_type_logits.shape:", action_type_logits.shape) if debug else None
    print("action_type:", action_type) if debug else None
    print("action_type.shape:", action_type.shape) if debug else None
    print("autoregressive_embedding:", autoregressive_embedding) if debug else None
    print("autoregressive_embedding.shape:", autoregressive_embedding.shape) if debug else None

    print("This is a test!") if debug else None


if __name__ == '__main__':
    test()
