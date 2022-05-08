#!/usr/bin/env python
# -*- coding: utf-8 -*-

" Selected Units Head."

import gc

import torch
import torch.nn as nn
import torch.nn.functional as F

from alphastarmini.lib import utils as L

from alphastarmini.lib.hyper_parameters import Arch_Hyper_Parameters as AHP
from alphastarmini.lib.hyper_parameters import StarCraft_Hyper_Parameters as SCHP
from alphastarmini.lib.hyper_parameters import Scalar_Feature_Size as SFS

import param as P

__author__ = "Ruo-Ze Liu"

debug = False


class SelectedUnitsHead(nn.Module):
    '''
    Inputs: autoregressive_embedding, action_type, entity_embeddings
    Outputs:
        units_logits - The logits corresponding to the probabilities of selecting each unit, 
            repeated for each of the possible 64 unit selections
        units - The units selected for this action.
        autoregressive_embedding - Embedding that combines information from `lstm_output` and all previous sampled arguments.
    '''

    def __init__(self, embedding_size=AHP.entity_embedding_size, 
                 max_number_of_unit_types=SCHP.max_unit_type, is_sl_training=True, 
                 temperature=AHP.temperature, max_selected=AHP.max_selected,
                 original_256=AHP.original_256, original_32=AHP.original_32,
                 autoregressive_embedding_size=AHP.autoregressive_embedding_size,
                 use_unit_type_entity_mask=AHP.use_unit_type_entity_mask):
        super().__init__()
        self.is_sl_training = is_sl_training
        self.temperature = temperature

        self.max_number_of_unit_types = max_number_of_unit_types
        self.func_embed = nn.Linear(max_number_of_unit_types, original_256)  # with relu

        self.conv_1 = nn.Conv1d(in_channels=embedding_size, 
                                out_channels=original_32, kernel_size=1, stride=1,
                                padding=0, bias=True)

        self.fc_1 = nn.Linear(autoregressive_embedding_size, original_256)
        self.fc_2 = nn.Linear(original_256, original_32)

        self.small_lstm = nn.LSTM(input_size=original_32, hidden_size=original_32, num_layers=1, 
                                  dropout=0.0, batch_first=True)

        self.max_selected = max_selected

        self.project = nn.Linear(original_32, autoregressive_embedding_size)
        self.softmax = nn.Softmax(dim=-1)

        # init a new tensor corrspond to end selection (also called EOF in NLP)
        # referenced by https://github.com/opendilab/DI-star in action_arg_head.py
        self.new_variable = nn.Parameter(torch.FloatTensor(original_32))
        nn.init.uniform_(self.new_variable, b=0.)
        #nn.init.uniform_(self.new_variable, b=1. / original_32)

        self.is_rl_training = False

        self.use_unit_type_entity_mask = use_unit_type_entity_mask

    def set_rl_training(self, staus):
        self.is_rl_training = staus

    def forward(self, autoregressive_embedding, action_type, entity_embeddings, entity_num, unit_type_entity_mask=None):
        '''
        Inputs:
            autoregressive_embedding: [batch_size x autoregressive_embedding_size]
            action_type: [batch_size x 1]
            entity_embeddings: [batch_size x entity_size x embedding_size]
            entity_num: [batch_size]
        Output:
            units_logits: [batch_size x max_selected x entity_size]
            units: [batch_size x max_selected x 1]
            autoregressive_embedding: [batch_size x autoregressive_embedding_size]
        '''
        batch_size = entity_embeddings.shape[0]
        entity_size = entity_embeddings.shape[-2]
        device = next(self.parameters()).device
        key_size = self.new_variable.shape[0]
        original_ae = autoregressive_embedding

        # AlphaStar: If applicable, Selected Units Head first determines which entity types can accept `action_type`,
        # creates a one-hot of that type with maximum equal to the number of unit types,
        # and passes it through a linear of size 256 and a ReLU. This will be referred to in this head as `func_embed`.
        # QUESTION: one unit type or serveral unit types?
        # ANSWER: serveral unit types, each for one-hot
        # This is some places which introduce much human knowledge
        unit_types_one_hot = L.action_can_apply_to_selected_mask(action_type).to(device)

        # the_func_embed shape: [batch_size x 256]
        the_func_embed = F.relu(self.func_embed(unit_types_one_hot))
        del unit_types_one_hot

        # AlphaStar: It also computes a mask of which units can be selected, initialised to allow selecting all entities 
        # that exist (including enemy units).
        # generate the length mask for all entities
        mask = torch.arange(entity_size, device=device).float()
        mask = mask.repeat(batch_size, 1)

        # now the entity nums should be added 1 (including the EOF)
        # this is because we also want to compute the mean including key value of the EOF
        added_entity_num = entity_num + 1

        # mask: [batch_size, entity_size]
        mask = mask < added_entity_num.unsqueeze(dim=1)
        assert mask.dtype == torch.bool

        # AlphaStar: It then computes a key corresponding to each entity by feeding `entity_embeddings`
        # through a 1D convolution with 32 channels and kernel size 1,
        # and creates a new variable corresponding to ending unit selection.
        # input: [batch_size x entity_size x embedding_size]
        # output: [batch_size x entity_size x key_size], note key_size = 32
        key = self.conv_1(entity_embeddings.transpose(-1, -2)).transpose(-1, -2)

        # end index should be the same to the entity_num
        end_index = entity_num

        # replace the EOF with the new_variable 
        # use calculation to achieve it
        if False:
            key[torch.arange(batch_size), end_index] = self.new_variable
        else:
            padding_end = torch.zeros(key.shape[0], 1, key.shape[2], dtype=key.dtype, device=key.device)
            key = torch.cat([key[:, :-1, :], padding_end], dim=1)

            flag = torch.ones(key.shape, dtype=torch.bool, device=key.device)
            flag[torch.arange(batch_size), end_index] = False

            # [batch_size, entity_size, key_size]
            end_embedding = torch.ones(key.shape, dtype=key.dtype, device=key.device) * self.new_variable.reshape(1, -1)
            key_end_part = end_embedding * ~flag

            # use calculation to replace new_variable
            key_main_part = key * flag
            key = key_main_part + key_end_part

            del padding_end, flag, end_embedding, key_main_part, key_end_part

        # calculate the average of keys (consider the entity_num)
        key_mask = mask.unsqueeze(dim=2).repeat(1, 1, key.shape[-1])
        key_avg = torch.sum(key * key_mask, dim=1) / entity_num.reshape(batch_size, 1)
        del key_mask

        # creates a new variable corresponding to ending unit selection.
        # QUESTION: how to do that?
        # ANSWER: referred by the DI-star project, please see self.new_variable in init() method
        units_logits = []
        units = []
        hidden = None

        # referneced by DI-star
        # represented which sample in the batch has end the selection
        # note is_end should be bool type to make sure it is a right whether mask 
        is_end = torch.zeros(batch_size, device=device).bool()

        # in the first selection, we should not select the end_index
        mask[torch.arange(batch_size), end_index] = False

        # if we stop selection early, we should record in each sample we select how many items
        select_units_num = torch.ones(batch_size, dtype=torch.long, device=device) * self.max_selected

        # AlphaStar: repeated for selecting up to 64 units
        for i in range(self.max_selected):
            if i == 1:
                mask[torch.arange(batch_size), end_index] = True  # in the second selection, we can select the EOF
                if self.is_rl_training and unit_type_entity_mask is not None:
                    unit_type_entity_mask[torch.arange(batch_size), end_index] = True

            x = self.fc_1(autoregressive_embedding)
            x = self.fc_2(F.relu(x + the_func_embed)).unsqueeze(dim=1)

            query, hidden = self.small_lstm(x, hidden)
            y = torch.sum(query * key, dim=-1)

            entity_logits = y.masked_fill(~mask, -1e9)
            if self.is_rl_training and self.use_unit_type_entity_mask and unit_type_entity_mask is not None:
                entity_logits = entity_logits.masked_fill(~unit_type_entity_mask, -1e9)

            temperature = self.temperature if self.is_rl_training else 1
            entity_logits = entity_logits / temperature
            del x, y, query

            entity_probs = self.softmax(entity_logits)
            entity_id = torch.multinomial(entity_probs, 1)

            units_logits.append(entity_logits.unsqueeze(-2))
            units.append(entity_id.unsqueeze(-2))

            mask[torch.arange(batch_size), entity_id.squeeze(dim=1)] = False  # masked out so that it cannot be selected in future iterations.

            last_index = (entity_id.squeeze(dim=1) == end_index)
            is_end[last_index] = 1

            # we record how many items we select in a sample
            # we select i + 1 items, but this include the EOF, so actually items should be i + 1 - 1
            select_units_num[last_index] = i

            # AlphaStar: The one-hot position of the selected entity is multiplied by the keys, 
            # reduced by the mean across the entities, passed through a linear layer of size 1024, 
            # and added to `autoregressive_embedding` for subsequent iterations. 
            entity_one_hot = L.tensor_one_hot(entity_id, entity_size).squeeze(-2)
            entity_one_hot_unsqueeze = entity_one_hot.unsqueeze(-2) 

            out = torch.bmm(entity_one_hot_unsqueeze, key).squeeze(-2)
            out = out - key_avg
            t = self.project(out)
            autoregressive_embedding = autoregressive_embedding + t * ~is_end.unsqueeze(dim=1)

            if P.skip_autoregressive_embedding:
                autoregressive_embedding = autoregressive_embedding - autoregressive_embedding
                autoregressive_embedding[:] = 0.

            del temperature, entity_logits, entity_probs, entity_id
            del last_index, entity_one_hot, entity_one_hot_unsqueeze, out, t

            if is_end.all():
                break

        # units_logits: [batch_size x select_units x entity_size]
        units_logits = torch.cat(units_logits, dim=1)

        # units: [batch_size x select_units x 1]
        units = torch.cat(units, dim=1)

        # we use zero padding to make units_logits has the size of [batch_size x max_selected x entity_size]
        # TODO: change the padding
        padding_size = self.max_selected - units_logits.shape[1]
        if padding_size > 0:
            pad_units_logits = torch.ones(units_logits.shape[0], padding_size, units_logits.shape[2],
                                          dtype=units_logits.dtype, device=units_logits.device) * (-1e9)
            units_logits = torch.cat([units_logits, pad_units_logits], dim=1) 

            pad_units = torch.zeros(units.shape[0], padding_size, units.shape[2],
                                    dtype=units.dtype, device=units.device)
            pad_units[:, :, 0] = entity_size - 1  # None index, the same as -1
            units = torch.cat([units, pad_units], dim=1)
            del pad_units, pad_units_logits

        # AlphaStar: If `action_type` does not involve selecting units, this head is ignored.

        # select_unit_mask: [batch_size x 1]
        # note select_unit_mask should be bool type to make sure it is a right whether mask 
        select_unit_mask = L.action_involve_selecting_units_mask(action_type).bool()

        no_select_units_index = ~select_unit_mask.squeeze(dim=1)
        print("no_select_units_index:", no_select_units_index) if debug else None

        select_units_num[no_select_units_index] = 0
        autoregressive_embedding[no_select_units_index] = original_ae[no_select_units_index]

        units_logits[no_select_units_index] = -1e9  # a magic number
        units[no_select_units_index, :, 0] = entity_size - 1  # None index, the same as -1

        print("select_units_num:", select_units_num) if debug else None
        print("autoregressive_embedding:", autoregressive_embedding) if debug else None

        del select_unit_mask, no_select_units_index, mask, is_end, key, key_avg

        return units_logits, units, autoregressive_embedding, select_units_num

    def mimic_forward(self, autoregressive_embedding, action_type, entity_embeddings, entity_num, units, select_units_num,
                      show=False, unit_type_entity_mask=None):
        '''
        Inputs:
            autoregressive_embedding: [batch_size x autoregressive_embedding_size]
            action_type: [batch_size x 1]
            entity_embeddings: [batch_size x entity_size x embedding_size]
            entity_num: [batch_size]
        Output:
            units_logits: [batch_size x max_selected x entity_size]
            units: [batch_size x max_selected x 1]
            autoregressive_embedding: [batch_size x autoregressive_embedding_size]
        '''
        batch_size = entity_embeddings.shape[0]
        entity_size = entity_embeddings.shape[-2]
        device = next(self.parameters()).device
        key_size = self.new_variable.shape[0]
        original_ae = autoregressive_embedding

        # AlphaStar: If applicable, Selected Units Head first determines which entity types can accept `action_type`,
        # creates a one-hot of that type with maximum equal to the number of unit types,
        # and passes it through a linear of size 256 and a ReLU. This will be referred to in this head as `func_embed`.
        # QUESTION: one unit type or serveral unit types?
        # ANSWER: serveral unit types, each for one-hot
        # This is some places which introduce much human knowledge
        unit_types_one_hot = L.action_can_apply_to_selected_mask(action_type).to(device)

        # the_func_embed shape: [batch_size x 256]
        the_func_embed = F.relu(self.func_embed(unit_types_one_hot))  
        del unit_types_one_hot

        # AlphaStar: It also computes a mask of which units can be selected, initialised to allow selecting all entities 
        # that exist (including enemy units).
        # generate the length mask for all entities
        mask = torch.arange(entity_size, device=device).float()
        mask = mask.repeat(batch_size, 1)

        # now the entity nums should be added 1 (including the EOF)
        # this is because we also want to compute the mean including key value of the EOF
        added_entity_num = entity_num + 1

        # mask: [batch_size, entity_size]
        mask = mask < added_entity_num.unsqueeze(dim=1)
        print("mask:", mask) if debug else None
        print("mask.shape:", mask.shape) if debug else None

        assert mask.dtype == torch.bool

        # AlphaStar: It then computes a key corresponding to each entity by feeding `entity_embeddings`
        # through a 1D convolution with 32 channels and kernel size 1,
        # and creates a new variable corresponding to ending unit selection.

        # input: [batch_size x entity_size x embedding_size]
        # output: [batch_size x entity_size x key_size], note key_size = 32
        key = self.conv_1(entity_embeddings.transpose(-1, -2)).transpose(-1, -2)

        # end index should be the same to the entity_num
        end_index = entity_num

        # replace the EOF with the new_variable 
        # use calculation to achieve it
        if False:
            key[torch.arange(batch_size), end_index] = self.new_variable
        else:
            padding_end = torch.zeros(key.shape[0], 1, key.shape[2], dtype=key.dtype, device=key.device)
            key = torch.cat([key[:, :-1, :], padding_end], dim=1)

            flag = torch.ones(key.shape, dtype=torch.bool, device=key.device)
            flag[torch.arange(batch_size), end_index] = False

            # [batch_size, entity_size, key_size]
            end_embedding = torch.ones(key.shape, dtype=key.dtype, device=key.device) * self.new_variable.reshape(1, -1)
            key_end_part = end_embedding * ~flag

            # use calculation to replace new_variable
            key_main_part = key * flag
            key = key_main_part + key_end_part
            del padding_end, flag, end_embedding, key_main_part, key_end_part

        # calculate the average of keys (consider the entity_num)
        key_mask = mask.unsqueeze(dim=2).repeat(1, 1, key.shape[-1])
        key_avg = torch.sum(key * key_mask, dim=1) / entity_num.reshape(batch_size, 1)
        del key_mask

        # creates a new variable corresponding to ending unit selection.
        # QUESTION: how to do that?
        # ANSWER: referred by the DI-star project, please see self.new_variable in init() method
        units_logits_list = []
        hidden = None

        # consider the EOF
        select_units_num = select_units_num + 1

        # designed with reference to DI-star
        max_seq_len = select_units_num.max()

        # for select_units_num
        selected_mask = torch.arange(max_seq_len, device=device).float()
        selected_mask = selected_mask.repeat(batch_size, 1)

        # mask: [batch_size, max_seq_len]
        selected_mask = selected_mask < select_units_num.unsqueeze(dim=1)
        assert selected_mask.dtype == torch.bool

        # in the first selection, we should not select the end_index
        mask[torch.arange(batch_size), end_index] = False

        is_end = torch.zeros(batch_size, device=device).bool()

        # designed with reference to DI-star
        for i in range(max_seq_len):
            if i != 0:
                # in the second selection, we can select the EOF
                if i == 1:
                    mask[torch.arange(batch_size), end_index] = True
                    if self.is_rl_training and unit_type_entity_mask is not None:
                        unit_type_entity_mask[torch.arange(batch_size), end_index] = True

            # AlphaStar: the network passes `autoregressive_embedding` through a linear of size 256,
            x = self.fc_1(autoregressive_embedding)

            # AlphaStar: adds `func_embed`, and passes the combination through a ReLU and a linear of size 32.
            # x shape: [batch_size x seq_len x 32], note seq_len = 1
            x = self.fc_2(F.relu(x + the_func_embed)).unsqueeze(dim=1)

            # AlphaStar: The result is fed into a LSTM with size 32 and zero initial state to get a query.
            query, hidden = self.small_lstm(x, hidden)
            y = torch.sum(query * key, dim=-1)

            # original mask usage is wrong, we should not let 0 * logits, zero value logit is still large! 
            # we use a very big negetive value replaced by logits, like -1e9
            # y shape: [batch_size x entity_size]
            entity_logits = y.masked_fill(~mask, -1e9)
            if self.is_rl_training and self.use_unit_type_entity_mask and unit_type_entity_mask is not None:
                entity_logits = entity_logits.masked_fill(~unit_type_entity_mask, -1e9)

            temperature = self.temperature if self.is_rl_training else 1
            entity_logits = entity_logits / temperature
            del x, y, query, temperature

            # note, we add a dimension where is in the seq_one to help
            # we concat to the one : [batch_size x max_selected x ?]
            units_logits_list.append(entity_logits.unsqueeze(-2))
            del entity_logits

            # the last EOF should not be considered
            if i != max_seq_len - 1:

                entity_id = units[:, i]
                print('entity_id', entity_id[0]) if show else None
                print('entity_id.shape', entity_id[0].shape) if show else None

                last_index = (entity_id.squeeze(dim=1) == end_index)
                is_end[last_index] = 1

                # AlphaStar: That entity is masked out so that it cannot be selected in future iterations.
                mask[torch.arange(batch_size), entity_id.squeeze(dim=1)] = False
                print('mask', mask[0]) if show else None

                # AlphaStar: The one-hot position of the selected entity is multiplied by the keys, 
                # reduced by the mean across the entities, passed through a linear layer of size 1024, 
                # and added to `autoregressive_embedding` for subsequent iterations. 
                entity_one_hot = L.tensor_one_hot(entity_id, entity_size).squeeze(-2)
                entity_one_hot_unsqueeze = entity_one_hot.unsqueeze(-2) 

                # entity_one_hot_unsqueeze shape: [batch_size x seq_len x entity_size], note seq_len =1 
                # key_shape: [batch_size x entity_size x key_size], note key_size = 32
                out = torch.bmm(entity_one_hot_unsqueeze, key).squeeze(-2)

                # AlphaStar: reduced by the mean across the entities,
                out = out - key_avg

                # t shape: [batch_size, autoregressive_embedding_size]
                t = self.project(out)

                # TODO, whether should be select_mask[:, i + 1] or select_mask[:, i] ?
                autoregressive_embedding = autoregressive_embedding + t * selected_mask[:, i + 1].unsqueeze(dim=1)
                if P.skip_autoregressive_embedding:
                    autoregressive_embedding = autoregressive_embedding - autoregressive_embedding
                    autoregressive_embedding[:] = 0.

                del t, out, entity_one_hot_unsqueeze, entity_one_hot, last_index, entity_id

                print("autoregressive_embedding:", autoregressive_embedding) if debug else None

        # in SL, we make the selected can have 1 more, like 12 + 1
        max_selected = self.max_selected + 1
        units_logits_size = len(units_logits_list)

        if units_logits_size >= max_selected:
            # remove the last one
            units_logits = torch.cat(units_logits_list[:max_selected], dim=1)
        elif units_logits_size > 0 and units_logits_size < max_selected:
            units_logits = torch.cat(units_logits_list, dim=1)
            padding_size = max_selected - units_logits.shape[1]
            if padding_size > 0:
                pad_units_logits = torch.ones(units_logits.shape[0], padding_size, units_logits.shape[2],
                                              dtype=units_logits.dtype, device=units_logits.device) * (-1e9)
                units_logits = torch.cat([units_logits, pad_units_logits], dim=1)
        else:
            units_logits = torch.ones(batch_size, max_selected, entity_size,
                                      dtype=action_type.dtype, device=action_type.device) * (-1e9)

        # AlphaStar: If `action_type` does not involve selecting units, this head is ignored.

        # select_unit_mask: [batch_size x 1]
        # note select_unit_mask should be bool type to make sure it is a right whether mask
        assert len(action_type.shape) == 2  

        select_unit_mask = L.action_involve_selecting_units_mask(action_type).bool()
        no_select_units_index = ~select_unit_mask.squeeze(dim=1)
        print("no_select_units_index:", no_select_units_index) if debug else None

        #autoregressive_embedding[no_select_units_index] = original_ae[no_select_units_index]
        units_logits[no_select_units_index] = (-1e9)  # a magic number

        # remove the EOF
        select_units_num = select_units_num - 1

        del selected_mask, select_unit_mask, no_select_units_index, mask, units_logits_list, key, key_avg

        return units_logits, None, autoregressive_embedding, select_units_num


def test():
    batch_size = 4
    autoregressive_embedding = torch.zeros(batch_size, AHP.autoregressive_embedding_size)
    action_type = torch.randint(low=0, high=SFS.available_actions, size=(batch_size, 1))
    action_type[0, 0] = 0  # no-op
    action_type[3, 0] = 168  # move-camera
    entity_embeddings = torch.randn(batch_size, AHP.max_entities, AHP.entity_embedding_size)
    entity_num = torch.tensor([1, 2, 3, 12])

    selected_units_head = SelectedUnitsHead()

    print("autoregressive_embedding:",
          autoregressive_embedding) if debug else None
    print("autoregressive_embedding.shape:",
          autoregressive_embedding.shape) if debug else None

    units_logits, units, autoregressive_embedding, units_num = \
        selected_units_head.forward(
            autoregressive_embedding, action_type, entity_embeddings, entity_num)

    if units_logits is not None:
        print("units_logits:", units_logits) if debug else None
        print("units_logits.shape:", units_logits.shape) if debug else None
    else:
        print("units_logits is None!")

    if units is not None:
        print("units:", units) if debug else None
        print("units.shape:", units.shape) if debug else None
    else:
        print("units is None!")

    print("units_num:", units_num) if debug else None

    units_logits, _, autoregressive_embedding, _ = \
        selected_units_head.mimic_forward(
            autoregressive_embedding, action_type, entity_embeddings, entity_num, units, units_num)

    if units_logits is not None:
        print("units_logits:", units_logits) if debug else None
        print("units_logits.shape:", units_logits.shape) if debug else None
    else:
        print("units_logits is None!")

    print("autoregressive_embedding:",
          autoregressive_embedding) if debug else None
    print("autoregressive_embedding.shape:",
          autoregressive_embedding.shape) if debug else None

    print("This is a test!") if debug else None


if __name__ == '__main__':
    test()
