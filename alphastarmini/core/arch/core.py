#!/usr/bin/env python
# -*- coding: utf-8 -*-

" Core."

import torch
import torch.nn as nn
import torch.nn.functional as F

from alphastarmini.lib.hyper_parameters import Arch_Hyper_Parameters as AHP

__author__ = "Ruo-Ze Liu"

debug = False


class Core(nn.Module):
    '''
    Inputs: prev_state, embedded_entity, embedded_spatial, embedded_scalar
    Outputs:
        next_state - The LSTM state for the next step
        lstm_output - The output of the LSTM
    '''

    def __init__(self, embedding_dim=AHP.original_1024, hidden_dim=AHP.lstm_hidden_dim, 
                 batch_size=AHP.batch_size,
                 sequence_length=AHP.sequence_length,
                 n_layers=AHP.lstm_layers, drop_prob=0.0):
        super(Core, self).__init__()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=n_layers, 
                            dropout=drop_prob, batch_first=True)
        self.batch_size = batch_size
        self.sequence_length = sequence_length

        #self.prev_state = None
        #self.dropout = nn.Dropout(drop_prob)
        #self.fc = nn.Linear(hidden_dim, output_size)
        #self.sigmoid = nn.Sigmoid()

    def forward(self, embedded_scalar, embedded_entity, embedded_spatial, 
                batch_size=None, sequence_length=None, hidden_state=None):
        # note: the input_shape[0] is batch_seq_size, we only transfrom it to [batch_size, seq_size, ...]
        # before input it into the lstm
        # shapes of embedded_entity, embedded_spatial, embedded_scalar are all [batch_seq_size x embedded_size]
        print('embedded_scalar.shape:', embedded_scalar.shape) if debug else None
        print('embedded_entity.shape:', embedded_entity.shape) if debug else None
        print('embedded_spatial.shape:', embedded_spatial.shape) if debug else None

        batch_seq_size = embedded_scalar.shape[0]
        print('batch_size:', batch_size) if debug else None
        print('self.batch_size:', self.batch_size) if debug else None
        batch_size = batch_size if batch_size is not None else self.batch_size
        sequence_length = sequence_length if sequence_length is not None else self.sequence_length

        print('batch_seq_size:', batch_seq_size) if debug else None
        print('batch_size:', batch_size) if debug else None
        print('sequence_length:', sequence_length) if debug else None

        assert batch_seq_size == batch_size * sequence_length
        assert batch_seq_size == embedded_entity.shape[0]
        assert batch_seq_size == embedded_spatial.shape[0]

        print('embedded_scalar is nan:', torch.isnan(embedded_scalar).any()) if debug else None
        print('embedded_entity is nan:', torch.isnan(embedded_entity).any()) if debug else None
        print('embedded_spatial is nan:', torch.isnan(embedded_spatial).any()) if debug else None

        input_tensor = torch.cat([embedded_scalar, embedded_entity, embedded_spatial], dim=-1)

        # note, before input to the LSTM
        # we transform the shape from [batch_seq_size, embedding_size] 
        # to the actual [batch_size, seq_size, embedding_size] 
        print('input_tensor.shape:', input_tensor.shape) if debug else None
        embedding_size = input_tensor.shape[-1]
        #input_tensor = input_tensor.unsqueeze(1)

        input_tensor = input_tensor.reshape(batch_size, sequence_length, embedding_size)
        print('input_tensor.shape:', input_tensor.shape) if debug else None

        print('input_tensor is nan:', torch.isnan(input_tensor).any()) if debug else None

        if hidden_state is None:
            hidden_state = self.init_hidden_state(batch_size=batch_size)

        lstm_output, hidden_state = self.forward_lstm(input_tensor, hidden_state)
        # lstm_output shape: [batch_size, seq_size, hidden_dim]
        print('lstm_output.shape:', lstm_output.shape) if debug else None
        # note, after the LSTM
        # we transform the shape from [batch_size, seq_size, hidden_dim] 
        # to the actual [batch_seq_size, hidden_dim] 

        lstm_output = lstm_output.reshape(batch_size * sequence_length, self.hidden_dim)
        return lstm_output, hidden_state

    def forward_lstm(self, x, hidden):
        # note: No projection is used.
        # note: The outputs of the LSTM are the outputs of this module.
        lstm_out, hidden = self.lstm(x, hidden)

        # DIFF: We apply layer norm to the gates.

        '''
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)
        
        out = self.dropout(lstm_out)
        out = self.fc(out)
        out = self.sigmoid(out)
        
        out = out.view(batch_size, -1)
        out = out[:,-1]
        '''

        return lstm_out, hidden

    def init_hidden_state(self, batch_size=1):
        '''
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device),
                  weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device))        
                  '''
        device = next(self.parameters()).device
        hidden = (torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(device), 
                  torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(device))
        return hidden


def test():

    print("This is a test!") if debug else None


if __name__ == '__main__':
    test()
