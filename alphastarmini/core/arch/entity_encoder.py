#!/usr/bin/env python
# -*- coding: utf-8 -*-

" Entity Encoder."

import time
import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pysc2.lib.units import Protoss, Neutral

from alphastarmini.lib.alphastar_transformer import Transformer
from alphastarmini.lib import utils as L

from alphastarmini.lib.hyper_parameters import Arch_Hyper_Parameters as AHP
from alphastarmini.lib.hyper_parameters import MiniStar_Arch_Hyper_Parameters as MAHP
from alphastarmini.lib.hyper_parameters import StarCraft_Hyper_Parameters as SCHP

__author__ = "Ruo-Ze Liu"

debug = False


class EntityEncoder(nn.Module):
    # below is value form max value of one-hot encoding
    max_entities = AHP.max_entities
    max_unit_type = SCHP.max_unit_type
    max_alliance = 5

    max_unit_attributes = 13

    max_health = 1500
    max_shield = 1000
    max_energy = 200

    max_cargo_space_used = 9
    max_cargo_space_maximum = 9

    max_display_type = 5
    max_cloakState = 5

    max_is_powered = 2
    max_is_hallucination = 2
    max_is_active = 2
    max_is_on_screen = 2
    max_is_in_cargo = 2

    max_current_minerals = 19
    max_current_vespene = 26

    max_mined_minerals = 1800
    max_mined_vespene = 2500

    max_assigned_harvesters = 25  # AlphaStar: 24. RuntimeError: index 24 is out of bounds for dimension 1 with size 24
    max_ideal_harvesters = 17

    max_weapon_cooldown = 32
    max_order_queue_length = 9

    max_order_progress = 10

    max_order_ids = SCHP.max_order_ids
    max_buffer_ids = SCHP.max_buffer_ids
    max_add_on_type = SCHP.max_add_on_type

    max_weapon_upgrades = 4
    max_armor_upgrades = 4
    max_shield_upgrades = 4

    max_was_selected = 2
    max_was_targeted = 2

    bias_value = -1e9

    '''
    Inputs: entity_list
    Outputs:
        embedded_entity - A 1D tensor of the embedded entities 
        entity_embeddings - The embedding of each entity (as opposed to `embedded_entity`, which has one embedding for all entities)
    '''

    def __init__(self, dropout=0.0, original_256=AHP.original_256, 
                 original_1024=AHP.original_1024,
                 original_128=AHP.original_128):
        super().__init__()

        self.dropout = nn.Dropout(dropout)
        self.embedd = nn.Linear(AHP.embedding_size, original_256)
        self.transformer = Transformer(d_model=original_256, d_inner=original_1024,
                                       n_layers=3, n_head=2, d_k=original_128, 
                                       d_v=original_128, 
                                       dropout=0.)  # make dropout=0 to make training and testing consistent
        self.conv1 = nn.Conv1d(original_256, original_256, kernel_size=1, stride=1,
                               padding=0, bias=True)
        self.fc1 = nn.Linear(original_256, original_256)

        # how many real entities we have
        self.real_entities_size = 0

    @classmethod
    def preprocess_numpy(cls, entity_list, return_entity_pos=False, debug=False):
        entity_array_list = []

        # add in mAS 1.05
        # the pos of each entity like (x, y)
        # note the x and y are in the position of minimap,
        # but the minimap size is not decided by the feature_dimensions(minimap), 
        # but the raw_resolution in AgentInterfaceFormatParams (It's a trap).
        # Be careful, if you give the raw_resolution a None, it will use the map default size.
        # E.g., if you use the AbyssalReef, give the feature_dimensions(minimap) a value of 64 but the 
        # raw_resolution a value of None. The raw_resolution will be about 152 x 132, so the (x, y) of
        # each entity will be much larger than 64.
        entity_pos_list = []

        index = 0
        for entity in entity_list:
            field_encoding_list = []

            # comments below have this style:
            # A: alphastar description
            # B: s2clientprotocol description
            # C: my notes

            # Note, from mAS 1.05, the comments have changed as following:

            # The following comments have this format:
            # 1. s2clientprotocol definition (s2clientprotocol/raw.proto)
            # 2. PySC2 (3.0) specfic preprocessing (full_unit_vec in features.py)
            # 3. AlphaStar processing (detailed-architecture.txt in pseudocode)
            # 4. mAS modification (entity_encoder.py)
            # The default order of data should be s2clientprotocol definition, PySC2 preprocessing, 
            # AlphaStar processing (and/or mAS modification).

            # -------------------------------------------------------------------------------
            # Start comments, the first is "unit_type"

            # A: unit_type: One-hot with maximum cls.max_unit_type (including unknown unit-type)
            # B: optional uint32 unit_type = 4;
            # C: with maximum cls.max_unit_type

            # 1. optional uint32 unit_type = 4;
            # 2. u.unit_type,
            # 3. unit_type: One-hot with maximum cls.max_unit_type (including unknown unit-type)
            # 4. with maximum cls.max_unit_typ
            unit_type = entity.unit_type

            # if unit_type == Protoss.Probe.value:
            #     debug = True
            # else:
            #     debug = False

            print('unit_type:', unit_type) if debug else None
            print('cls.max_unit_type:', cls.max_unit_type) if debug else None

            # note: if we use the SC2 built-in unit type ID, due to its max > 2000, is costs many space for one-hot,
            # on the contray, we only have about 259 unit types, so we change the built-in unit type to our version
            unit_type_index = L.unit_tpye_to_unit_type_index(unit_type)
            print('unit_type_index:', unit_type_index) if debug else None
            assert unit_type_index >= 0 and unit_type_index <= cls.max_unit_type

            unit_type_encoding = L.np_one_hot(np.array([unit_type_index]), cls.max_unit_type)
            print('unit_type_encoding:', unit_type_encoding) if debug else None
            print('unit_type_encoding.shape:', unit_type_encoding.shape) if debug else None
            field_encoding_list.append(unit_type_encoding)

            # A: unit_attributes: One boolean for each of the 13 unit attributes
            # B: not found
            # C: lack

            # 1. not found
            # 2. doesn't know which the "13 unit attributes" refer to
            # 3. unit_attributes: One boolean for each of the 13 unit attributes
            # 4. lack
            unit_attributes_encoding = np.array(entity.unit_attributes, dtype=np.float32).reshape(1, -1)
            print('unit_attributes_encoding:', unit_attributes_encoding) if debug else None
            print('unit_attributes_encoding.shape:', unit_attributes_encoding.shape) if debug else None
            field_encoding_list.append(unit_attributes_encoding)

            # A: alliance: One-hot with maximum 5 (including unknown alliance)
            # B: optional Alliance alliance = 2; not max is 4, not 5
            # C: use A

            # 1. optional Alliance alliance = 2; note type of Alliance is enum in proto
            # 2. u.alliance,  # Self = 1, Ally = 2, Neutral = 3, Enemy = 4, so considering 0, we have 5 types
            # 3. alliance: One-hot with maximum 5 (including unknown alliance)
            # 4. the same as AlphaStar
            alliance_encoding = L.np_one_hot(np.array([entity.alliance]), cls.max_alliance)
            print('alliance_encoding:', alliance_encoding) if debug else None
            print('alliance_encoding.shape:', alliance_encoding.shape) if debug else None
            field_encoding_list.append(alliance_encoding)

            # A: display_type: One-hot with maximum 5
            # B: note: in s2clientprotocol raw.proto, display type only has 4 values, type of enum DisplayType,
            # C: we keep in consistent with s2clientprotocol

            # 1. enum DisplayType, Snapshot means Dimmed version of unit left behind after entering fog of war, 
            # 2. u.display_type,  # Visible = 1, Snapshot = 2, Hidden = 3 (Placeholder = 4 in sc2clientprotocal)
            # 3. display_type: One-hot with maximum 5
            # 4. the same as AlphaStar
            display_type_encoding = L.np_one_hot(np.array([entity.display_type]), cls.max_display_type)
            print('display_type_encoding:', display_type_encoding) if debug else None
            print('display_type_encoding.shape:', display_type_encoding.shape) if debug else None
            field_encoding_list.append(display_type_encoding)

            # A: x_position: Binary encoding of entity x-coordinate, in game units
            # B: optional Point pos = 6;
            # C: use np.unpackbits

            # 1. x of Point pos; Point on the map world from 0 to 255 (maxsize of map is 256). bottom left is 0, 0.
            # 2. screen_pos.x, note "screen_pos" is transfromed pixel in screen. If using raw data, it is Minimap!
            # 3. x_position: Binary encoding of entity x-coordinate, in game units
            # 4. use np.unpackbits to do Binary encoding. Note if using raw=True, it is the postion of minimap.          
            x_encoding = np.unpackbits(np.array([entity.x], np.uint8)).reshape(1, -1)
            print('x_encoding:', x_encoding) if debug else None
            print('x_encoding.shape:', x_encoding.shape) if debug else None
            field_encoding_list.append(x_encoding)

            # A: y_position: Binary encoding of entity y-coordinate, in game units
            # B: optional Point pos = 6;
            # C: use np.unpackbits
            # change sequence due to easy processing of entity_x_y_index

            # 1. y of Point pos; Point on the map world from 0 to 255 (maxsize of map is 256). 
            # 2. screen_pos.y, note "screen_pos" is transfromed pixel in Minimap if using raw=True!
            # 3. y_position: Binary encoding of entity x-coordinate, in game units
            # 4. use np.unpackbits to do Binary encoding. Note if using raw=True, it is the postion of minimap.             
            y_encoding = np.unpackbits(np.array([entity.y], np.uint8)).reshape(1, -1)
            print('y_encoding:', y_encoding) if debug else None
            print('y_encoding.shape:', y_encoding.shape) if debug else None
            field_encoding_list.append(y_encoding)

            # for use in the spatial encoder's to calculate the scatter map.
            entity_pos_list.append([entity.x, entity.y])

            # A: current_health: One-hot of sqrt(min(current_health, 1500)) with maximum sqrt(1500), rounding down
            # B: optional float health = 14;
            # C: None

            # 1. optional float health = 14;
            # 2. u.health
            # 3. current_health: One-hot of sqrt(min(current_health, 1500)) with maximum sqrt(1500), rounding down
            # 4. The same as AlphaStar            
            print('entity.health:', entity.health) if debug else None
            current_health = int(min(entity.health, cls.max_health) ** 0.5)
            print('current_health:', current_health) if debug else None
            current_health_encoding = L.np_one_hot(np.array([current_health]), int(cls.max_health ** 0.5) + 1)
            print('current_health_encoding:', current_health_encoding) if debug else None
            print('current_health_encoding.shape:', current_health_encoding.shape) if debug else None
            field_encoding_list.append(current_health_encoding)

            # A: current_shields: One-hot of sqrt(min(current_shields, 1000)) with maximum sqrt(1000), rounding down
            # B: optional float shield = 16;
            # C: None

            # 1. optional float shield = 16;
            # 2. u.shield,
            # 3. current_shields: One-hot of sqrt(min(current_shields, 1000)) with maximum sqrt(1000), rounding down
            # 4. The same as AlphaStar
            print('entity.shield:', entity.shield) if debug else None
            current_shield = int(min(entity.shield, cls.max_shield) ** 0.5)
            print('current_shield:', current_shield) if debug else None
            current_shield_encoding = L.np_one_hot(np.array([current_shield]), int(cls.max_shield ** 0.5) + 1)
            print('current_shield_encoding:', current_shield_encoding) if debug else None
            print('current_shield_encoding.shape:', current_shield_encoding.shape) if debug else None
            field_encoding_list.append(current_shield_encoding)

            # A: current_energy: One-hot of sqrt(min(current_energy, 200)) with maximum sqrt(200), rounding down
            # B: optional float energy = 17;
            # C: None

            # 1. optional float energy = 17;
            # 2. u.energy,
            # 3. current_energy: One-hot of sqrt(min(current_energy, 200)) with maximum sqrt(200), rounding down
            # 4. The same as AlphaStar            
            print('entity.energy:', entity.energy) if debug else None
            current_energy = int(min(entity.energy, cls.max_energy) ** 0.5)
            print('current_energy:', current_energy) if debug else None
            current_energy_encoding = L.np_one_hot(np.array([current_energy]), int(cls.max_energy ** 0.5) + 1)
            print('current_energy_encoding:', current_energy_encoding) if debug else None
            print('current_energy_encoding.shape:', current_energy_encoding.shape) if debug else None
            field_encoding_list.append(current_energy_encoding)

            # A: cargo_space_used: One-hot with maximum 9
            # B: optional int32 cargo_space_taken = 25;
            # C: None

            # 1. optional int32 cargo_space_taken = 25; // Not populated for enemies
            # 2. u.cargo_space_taken,
            # 3. cargo_space_used: One-hot with maximum 9
            # 4. The same as AlphaStar  
            cargo_space_used = entity.cargo_space_taken
            assert cargo_space_used >= 0 and cargo_space_used <= 8
            cargo_space_used_encoding = L.np_one_hot(np.array([cargo_space_used]), cls.max_cargo_space_used)
            print('cargo_space_used_encoding:', cargo_space_used_encoding) if debug else None
            field_encoding_list.append(cargo_space_used_encoding)

            # A: cargo_space_maximum: One-hot with maximum 9
            # B: optional int32 cargo_space_max = 26;
            # C: None

            # 1. optional int32 cargo_space_max = 26; // Not populated for enemies
            # 2. u.cargo_space_max, # Not populated for enemies or neutral
            # 3. cargo_space_maximum: One-hot with maximum 9
            # 4. The same as AlphaStar          
            cargo_space_maximum = entity.cargo_space_max
            assert cargo_space_maximum >= 0 and cargo_space_maximum <= 8
            cargo_space_maximum_encoding = L.np_one_hot(np.array([cargo_space_maximum]), cls.max_cargo_space_maximum)
            print('cargo_space_maximum_encoding:', cargo_space_maximum_encoding) if debug else None
            field_encoding_list.append(cargo_space_maximum_encoding)

            # A: build_progress: Float of build progress, in [0, 1]
            # B: optional float build_progress = 9;        // Range: [0.0, 1.0]
            # C: None

            # 1. optional float build_progress = 9;        // Range: [0.0, 1.0]
            # 2. int(u.build_progress * 100),  # discretize
            # 3. build_progress: Float of build progress, in [0, 1]
            # 4. The same as AlphaStar. Note in mAS before 1.05 version, it is wrong like 100 not 1.0  
            print('entity.build_progress:', entity.build_progress) if debug else None
            build_progress = entity.build_progress / 100.
            print('build_progress:', build_progress) if debug else None
            build_progress_encoding = np.array([build_progress], dtype=np.float32).reshape(1, -1)
            print('build_progress_encoding:', build_progress_encoding) if debug else None
            print('build_progress_encoding.shape:', build_progress_encoding.shape) if debug else None
            field_encoding_list.append(build_progress_encoding)

            # A: current_health_ratio: Float of health ratio, in [0, 1]
            # B: optional float health_max = 15;
            # C: None

            # 1. optional float health_max = 15; // raw.proto doesn't have a health_ratio property
            # 2. health_ratio = 7. int(u.health / u.health_max * 255) if u.health_max > 0 else 0,
            # 3. current_health_ratio: Float of health ratio, in [0, 1]
            # 4. The same as AlphaStar. Note in mAS before 1.05 version, it isn't calculated right.
            health_ratio = entity.current_health_ratio / 255.
            print('health_ratio:', health_ratio) if debug else None  
            current_health_ratio_encoding = np.array([health_ratio], dtype=np.float32).reshape(1, -1)
            print('current_health_ratio_encoding:', current_health_ratio_encoding) if debug else None
            field_encoding_list.append(current_health_ratio_encoding)

            # A: current_shield_ratio: Float of shield ratio, in [0, 1]
            # B: optional float shield_max = 36;
            # C: None

            # 1. optional float shield_max = 36; // raw.proto doesn't have a shield_ratio property
            # 2. shield_ratio = 8. int(u.shield / u.shield_max * 255) if u.shield_max > 0 else 0,
            # 3. current_shield_ratio: Float of shield ratio, in [0, 1]
            # 4. The same as AlphaStar. Note in mAS before 1.05 version, it isn't calculated right.
            print('entity.current_shield_ratio:', entity.current_shield_ratio) if debug else None 
            # Note, in 3.16.1 SC2, the current_shield_ratio returns is zero
            shield_ratio = entity.current_shield_ratio / 255.
            print('shield_ratio:', shield_ratio) if debug else None  
            current_shield_ratio_encoding = np.array([shield_ratio], dtype=np.float32).reshape(1, -1)
            print('current_shield_ratio_encoding:', current_shield_ratio_encoding) if debug else None
            field_encoding_list.append(current_shield_ratio_encoding)

            # A: current_energy_ratio: Float of energy ratio, in [0, 1]
            # B: optional float energy_max = 37;
            # C: None

            # 1. optional float energy_max = 36; // raw.proto doesn't have a energy_ratio property
            # 2. energy_ratio = 9. int(u.energy / u.energy_max * 255) if u.energy_max > 0 else 0,
            # 3. current_energy_ratio: Float of energy ratio, in [0, 1]
            # 4. The same as AlphaStar. Note in mAS before 1.05 version, it isn't calculated right.
            energy_ratio = entity.current_energy_ratio / 255.
            current_energy_ratio_encoding = np.array([energy_ratio], dtype=np.float32).reshape(1, -1)
            print('current_energy_ratio_encoding:', current_energy_ratio_encoding) if debug else None
            field_encoding_list.append(current_energy_ratio_encoding)

            # A: is_cloaked: One-hot with maximum 5
            # B: note: in s2clientprotocol raw.proto, this is called clock, type of enum CloakState,
            # C: we keep in consistent with s2clientprotocol
            # clock_value = entity.cloak.value

            # 1. CloakState cloak = 10. CloakedUnknown = 0;  // Under the fog, so unknown whether it's cloaked or not.
            # 2. index of cloak = 16. u.cloak,  # Cloaked = 1, CloakedDetected = 2, NotCloaked = 3
            # 3. is_cloaked: One-hot with maximum 5
            # 4. e.is_cloaked = raw_unit.cloak
            cloakState_encoding = L.np_one_hot(np.array([entity.is_cloaked]), cls.max_cloakState)
            print('cloakState_encoding:', cloakState_encoding) if debug else None
            field_encoding_list.append(cloakState_encoding)

            # A: is_powered: One-hot with maximum 2
            # B: note: in s2clientprotocol raw.proto, this is type of bool
            # C: we convert it from bool to int

            # 1. note: in s2clientprotocol raw.proto, this is type of bool
            # 2. index of is_powered = 19. u.is_powered,
            # 3. is_powered: One-hot with maximum 2
            # 4. we convert it from bool to int            
            print('entity.is_powered:', entity.is_powered) if debug else None
            is_powered_value = int(entity.is_powered)
            print('is_powered_value:', is_powered_value) if debug else None
            is_powered_encoding = L.np_one_hot(np.array([is_powered_value]), cls.max_is_powered)
            print('is_powered_encoding:', is_powered_encoding) if debug else None
            field_encoding_list.append(is_powered_encoding)

            # A: is_hallucination: One-hot with maximum 2
            # B: note: in s2clientprotocol raw.proto, this is type of bool
            # C: we convert it from bool to int

            # 1. note: in s2clientprotocol raw.proto, this is type of bool
            # 2. index of hallucination = 30. u.is_hallucination,
            # 3. is_hallucination: One-hot with maximum 2
            # 4. we convert it from bool to int  
            print('entity.is_hallucination:', entity.is_hallucination) if debug else None
            is_hallucination_value = int(entity.is_hallucination)
            print('is_hallucination_value:', is_hallucination_value) if debug else None
            is_hallucination_encoding = L.np_one_hot(np.array([is_hallucination_value]), cls.max_is_hallucination)
            print('is_hallucination_encoding:', is_hallucination_encoding) if debug else None
            field_encoding_list.append(is_hallucination_encoding)

            # A: is_active: One-hot with maximum 2
            # B: note: in s2clientprotocol raw.proto, this is type of bool
            # C: we convert it from bool to int

            # 1. optional bool is_active = 39; // Building is training/researching (ie animated).
            # 2. index of active = 34. u.is_active,
            # 3. is_active: One-hot with maximum 2
            # 4. we convert it from bool to int 
            is_active_value = int(entity.is_active)
            is_active_encoding = L.np_one_hot(np.array([is_active_value]), cls.max_is_active)
            print('is_active_encoding:', is_active_encoding) if debug else None
            field_encoding_list.append(is_active_encoding)

            # A: is_on_screen: One-hot with maximum 2
            # B: note: in s2clientprotocol raw.proto, this is type of bool
            # C: we convert it from bool to int

            # 1. optional bool is_on_screen = 12;  // Visible and within the camera frustrum.
            # 2. index of is_on_screen = 35. u.is_on_screen,
            # 3. is_on_screen: One-hot with maximum 2
            # 4. we convert it from bool to int 
            is_on_screen_value = int(entity.is_on_screen)
            is_on_screen_encoding = L.np_one_hot(np.array([is_on_screen_value]), cls.max_is_on_screen)
            print('is_on_screen_encoding:', is_on_screen_encoding) if debug else None
            field_encoding_list.append(is_on_screen_encoding)

            # A: is_in_cargo: One-hot with maximum 2
            # B: note: in s2clientprotocol raw.proto, there is no is_in_cargo
            # C: wait to be resolved by other ways

            # 1. in s2clientprotocol raw.proto, there is no is_in_cargo
            # 2. index of is_in_cargo = 40. always 0,
            # 3. is_in_cargo: One-hot with maximum 2
            # 4. None            
            is_in_cargo_value = int(entity.is_in_cargo)
            is_in_cargo_encoding = L.np_one_hot(np.array([is_in_cargo_value]), cls.max_is_in_cargo)
            print('is_in_cargo_encoding:', is_in_cargo_encoding) if debug else None
            field_encoding_list.append(is_in_cargo_encoding)

            # A: current_minerals: One-hot of (current_minerals / 100) with maximum 19, rounding down
            # B: optional int32 mineral_contents = 18; (maybe)
            # C: I am not sure mineral_contents corrseponds to current_minerals
            print('entity.current_minerals:', entity.current_minerals) if debug else None
            current_minerals = int(entity.current_minerals / 100)
            print('current_minerals:', current_minerals) if debug else None
            current_minerals_encoding = L.np_one_hot(np.array([current_minerals]), cls.max_current_minerals).reshape(1, -1)
            print('current_minerals_encoding.shape:', current_minerals_encoding.shape) if debug else None
            field_encoding_list.append(current_minerals_encoding)

            # A: current_vespene: One-hot of (current_vespene / 100) with maximum 26, rounding down
            # B: optional int32 vespene_contents = 19; (maybe)
            # C: I am not sure vespene_contents corrseponds to current_vespene
            print('entity.current_vespene:', entity.current_vespene) if debug else None
            current_vespene = int(entity.current_vespene / 100)
            print('current_vespene:', current_vespene) if debug else None
            current_vespene_encoding = L.np_one_hot(np.array([current_vespene]), cls.max_current_vespene).reshape(1, -1)
            print('current_vespene_encoding.shape:', current_vespene_encoding.shape) if debug else None
            field_encoding_list.append(current_vespene_encoding)

            # A: mined_minerals: One-hot of sqrt(min(mined_minerals, 1800)) with maximum sqrt(1800), rounding down
            # B: not found
            # C: wait to be resolved by other ways
            print('entity.mined_minerals:', entity.mined_minerals) if debug else None
            mined_minerals = int(min(entity.mined_minerals, cls.max_mined_minerals) ** 0.5)
            print('mined_minerals:', mined_minerals) if debug else None
            mined_minerals_encoding = L.np_one_hot(np.array([mined_minerals]), int(cls.max_mined_minerals ** 0.5) + 1).reshape(1, -1)
            print('mined_minerals_encoding.shape:', mined_minerals_encoding.shape) if debug else None
            field_encoding_list.append(mined_minerals_encoding)

            # A: mined_vespene: One-hot of sqrt(min(mined_vespene, 2500)) with maximum sqrt(2500), rounding down
            # B: not found
            # C: wait to be resolved by other ways
            print('entity.mined_vespene:', entity.mined_vespene) if debug else None
            mined_vespene = int(min(entity.mined_vespene, cls.max_mined_vespene) ** 0.5)
            print('mined_vespene:', mined_vespene) if debug else None
            mined_vespene_encoding = L.np_one_hot(np.array([mined_vespene]), int(cls.max_mined_vespene ** 0.5) + 1).reshape(1, -1)
            print('mined_vespene_encoding.shape:', mined_vespene_encoding.shape) if debug else None
            field_encoding_list.append(mined_vespene_encoding)

            # A: assigned_harvesters: One-hot with maximum 24
            # B: optional int32 assigned_harvesters = 28;
            # C: None
            assigned_harvesters_encoding = L.np_one_hot(np.array([min(entity.assigned_harvesters, 24)]), cls.max_assigned_harvesters).reshape(1, -1)
            print('assigned_harvesters_encoding:', assigned_harvesters_encoding) if debug else None
            field_encoding_list.append(assigned_harvesters_encoding)

            # A: ideal_harvesters: One-hot with maximum 17
            # B: optional int32 ideal_harvesters = 29;
            # C: None
            ideal_harvesters_encoding = L.np_one_hot(np.array([entity.ideal_harvesters]), cls.max_ideal_harvesters).reshape(1, -1)
            print('ideal_harvesters_encoding:', ideal_harvesters_encoding) if debug else None
            field_encoding_list.append(ideal_harvesters_encoding)

            # A: weapon_cooldown: One-hot with maximum 32 (counted in game steps)
            # B: optional float weapon_cooldown = 30;
            # C: None
            weapon_cooldown = int(entity.weapon_cooldown)
            weapon_cooldown = min(weapon_cooldown, 31)
            weapon_cooldown_encoding = L.np_one_hot(np.array([weapon_cooldown]), cls.max_weapon_cooldown).reshape(1, -1)
            print('weapon_cooldown_encoding:', weapon_cooldown_encoding) if debug else None
            field_encoding_list.append(weapon_cooldown_encoding)

            # A: order_queue_length: One-hot with maximum 9
            # B: repeated UnitOrder orders = 22; Not populated for enemies;
            # C: equal to FeatureUnit.order_length
            order_queue_length = entity.order_length
            order_queue_length = min(order_queue_length, 8)
            order_queue_length_encoding = L.np_one_hot(np.array([order_queue_length]), cls.max_order_queue_length).reshape(1, -1)
            print('order_queue_length_encoding:', order_queue_length_encoding) if debug else None
            field_encoding_list.append(order_queue_length_encoding)

            # A: order_1: One-hot across all order IDs
            # B: below is the definition of order
            '''
                message UnitOrder {
                      optional uint32 ability_id = 1;
                      oneof target {
                        Point target_world_space_pos = 2;
                        uint64 target_unit_tag = 3;
                      }
                      optional float progress = 4;              // Progress of train abilities. Range: [0.0, 1.0]
                    }
            '''
            # C: actually this is across all ability_ids in orders, lack: a vector for all ability_ids
            order_1 = entity.order_id_1
            order_1_encoding = L.np_one_hot(np.array([order_1]), cls.max_order_ids).reshape(1, -1)
            print('order_1_encoding:', order_1_encoding) if debug else None
            field_encoding_list.append(order_1_encoding)

            # A: order_2: One-hot across all building training order IDs. Note that this tracks queued building orders, and unit orders will be ignored
            # B: None
            # C: in mAS, we ingore it
            if AHP != MAHP:
                order_2 = entity.order_id_2
                order_2_encoding = L.np_one_hot(np.array([order_2]), cls.max_order_ids).reshape(1, -1)
                print('order_2_encoding:', order_2_encoding) if debug else None
                field_encoding_list.append(order_2_encoding)

            # A: order_3: One-hot across all building training order IDs
            # B: None
            # C: in mAS, we ingore it
            if AHP != MAHP:
                order_3 = entity.order_id_3
                order_3_encoding = L.np_one_hot(np.array([order_3]), cls.max_order_ids).reshape(1, -1)
                print('order_3_encoding:', order_3_encoding) if debug else None
                field_encoding_list.append(order_3_encoding)

            # A: order_4: One-hot across all building training order IDs
            # B: None
            # C: in mAS, we ingore it
            if AHP != MAHP:
                order_4 = entity.order_id_4
                order_4_encoding = L.np_one_hot(np.array([order_4]), cls.max_order_ids).reshape(1, -1)
                print('order_4_encoding:', order_4_encoding) if debug else None
                field_encoding_list.append(order_4_encoding)

            # A: buffs: Boolean for each buff of whether or not it is active. Only the first two buffs are tracked
            # B: None
            # C: in mAS, we ingore buff_id_2
            buff_id_1 = entity.buff_id_1
            buff_id_1_encoding = L.np_one_hot(np.array([buff_id_1]), cls.max_buffer_ids).reshape(1, -1)
            print('buff_id_1_encoding:', buff_id_1_encoding) if debug else None
            field_encoding_list.append(buff_id_1_encoding)            

            if AHP != MAHP:
                buff_id_2 = entity.buff_id_2
                buff_id_2_encoding = L.np_one_hot(np.array([buff_id_2]), cls.max_buffer_ids).reshape(1, -1)
                print('buff_id_2_encoding:', buff_id_2_encoding) if debug else None
                field_encoding_list.append(buff_id_2_encoding)

            # A: addon_type: One-hot of every possible add-on type
            # B: optional uint64 add_on_tag = 23;
            # C: lack: from tag to find a unit, then get the unit type of the unit 
            # in mAS, we ingore it

            # TODO
            if AHP != MAHP:
                addon_unit_type = entity.addon_unit_type
                addon_unit_type_encoding = L.np_one_hot(np.array([addon_unit_type]), cls.max_add_on_type).reshape(1, -1)
                print('addon_unit_type_encoding:', addon_unit_type_encoding) if debug else None
                field_encoding_list.append(addon_unit_type_encoding)

            # A: order_progress_1: Float of order progress, in [0, 1], and one-hot of (`order_progress_1` / 10) with maximum 10
            # B: optional float progress = 4;              // Progress of train abilities. Range: [0.0, 1.0]
            # C: None
            '''
            orders = entity.orders
            if len(orders) >= 1:
                order_1 = orders[0]
                if hasattr(order_1, 'progress'):
            '''

            order_progress_1_encoding = np.zeros((1, 1), dtype=np.float32)
            order_progress_1_encoding_2 = np.zeros((1, cls.max_order_progress), dtype=np.float32)
            order_progress_1 = entity.order_progress_1
            if order_progress_1 is not None:
                order_progress_1_encoding = np.array([order_progress_1 / 100.], dtype=np.float32).reshape(1, -1)
                print('order_progress_1', order_progress_1) if debug else None
                print('cls.max_order_progress', cls.max_order_progress) if debug else None
                order_progress_1_encoding_2 = L.np_one_hot(np.array([int(order_progress_1 / 10)]),
                                                           cls.max_order_progress).reshape(1, -1)
            print('order_progress_1_encoding:', order_progress_1_encoding) if debug else None
            field_encoding_list.append(order_progress_1_encoding)
            print('order_progress_1_encoding_2:', order_progress_1_encoding_2) if debug else None
            field_encoding_list.append(order_progress_1_encoding_2)

            # A: order_progress_2: Float of order progress, in [0, 1], and one-hot of (`order_progress_2` / 10) with maximum 10
            # B: optional float progress = 4;              // Progress of train abilities. Range: [0.0, 1.0]
            # C: None
            '''
            if len(orders) >= 2:
                order_2 = orders[1]
                if hasattr(order_2, 'progress'):
            '''

            order_progress_2_encoding = np.zeros((1, 1), dtype=np.float32)
            order_progress_2_encoding_2 = np.zeros((1, cls.max_order_progress), dtype=np.float32)
            order_progress_2 = entity.order_progress_2
            if order_progress_2 is not None:
                order_progress_2_encoding = np.array([order_progress_2 / 100.], dtype=np.float32).reshape(1, -1)
                order_progress_2_encoding_2 = L.np_one_hot(np.array([int(order_progress_2 / 10)]),
                                                           cls.max_order_progress).reshape(1, -1)
            print('order_progress_2_encoding:', order_progress_2_encoding) if debug else None
            field_encoding_list.append(order_progress_2_encoding)
            print('order_progress_2_encoding_2:', order_progress_2_encoding_2) if debug else None
            field_encoding_list.append(order_progress_2_encoding_2)

            # A: weapon_upgrades: One-hot with maximum 4
            # B: optional int32 attack_upgrade_level = 40;
            # C: None
            weapon_upgrades = entity.attack_upgrade_level
            assert weapon_upgrades >= 0 and weapon_upgrades <= 3
            weapon_upgrades_encoding = L.np_one_hot(np.array([weapon_upgrades]), cls.max_weapon_upgrades).reshape(1, -1)
            print('weapon_upgrades_encoding:', weapon_upgrades_encoding) if debug else None
            field_encoding_list.append(weapon_upgrades_encoding)

            # A: armor_upgrades: One-hot with maximum 4
            # B: optional int32 armor_upgrade_level = 41;
            # C: None
            armor_upgrades = entity.armor_upgrade_level
            assert armor_upgrades >= 0 and armor_upgrades <= 3
            armor_upgrades_encoding = L.np_one_hot(np.array([armor_upgrades]), cls.max_armor_upgrades).reshape(1, -1)
            print('armor_upgrades_encoding:', armor_upgrades_encoding) if debug else None
            field_encoding_list.append(armor_upgrades_encoding)

            # A: shield_upgrades: One-hot with maximum 4
            # B: optional int32 armor_upgrade_level = 41;
            # C: None
            shield_upgrades = entity.shield_upgrade_level
            assert shield_upgrades >= 0 and shield_upgrades <= 3
            shield_upgrades_encoding = L.np_one_hot(np.array([shield_upgrades]), cls.max_shield_upgrades).reshape(1, -1)
            print('shield_upgrades_encoding:', shield_upgrades_encoding) if debug else None
            field_encoding_list.append(shield_upgrades_encoding)

            # A: was_selected: One-hot with maximum 2 of whether this unit was selected last actionn
            # B: optional bool is_selected = 11;
            # C: None
            was_selected = int(entity.is_selected)
            was_selected_encoding = L.np_one_hot(np.array([was_selected]), cls.max_was_selected).reshape(1, -1)
            print('was_selected_encoding:', was_selected_encoding) if debug else None
            field_encoding_list.append(was_selected_encoding)

            # A: was_targeted: One-hot with maximum 2 of whether this unit was targeted last action
            # B: not found,   optional uint64 engaged_target_tag = 34; (maybe)
            # C: None
            was_targeted = int(entity.is_targeted)
            was_targeted_encoding = L.np_one_hot(np.array([was_targeted]), cls.max_was_targeted).reshape(1, -1)
            print('was_targeted_encoding:', was_targeted_encoding) if debug else None
            field_encoding_list.append(was_targeted_encoding)

            entity_array = np.concatenate(field_encoding_list, axis=1)
            print('entity_array:', entity_array) if debug else None
            print('entity_array.shape:', entity_array.shape) if debug else None

            # There are up to 512 of these preprocessed entities, and any entities after 512 are ignored.
            if index < cls.max_entities:
                entity_array_list.append(entity_array)
            else:
                break
            index = index + 1

        all_entities_array = np.concatenate(entity_array_list, axis=0)

        # count how many real entities we have
        real_entities_size = all_entities_array.shape[0]

        # we use a bias of -1e9 for any of the 512 entries that doesn't refer to an entity.
        # TODO: make it better
        if all_entities_array.shape[0] < cls.max_entities:
            bias_length = cls.max_entities - all_entities_array.shape[0]
            bias = np.zeros((bias_length, AHP.embedding_size))
            bias[:, :] = cls.bias_value
            all_entities_array = np.concatenate([all_entities_array, bias], axis=0)

        if return_entity_pos:
            return all_entities_array, entity_pos_list

        return all_entities_array

    def forward(self, x, debug=True):
        # refactor thanks mostly to the codes from https://github.com/opendilab/DI-star
        batch_size = x.shape[0]

        # calculate there are how many real entities in each batch
        # tmp_x: [batch_seq_size x entities_size]
        tmp_x = torch.mean(x, dim=2, keepdim=False)

        # tmp_y: [batch_seq_size x entities_size]
        tmp_y = (tmp_x > self.bias_value + 1e3)

        # entity_num: [batch_seq_size]
        entity_num = torch.sum(tmp_y, dim=1, keepdim=False)

        # this means for each batch, there are how many real enetities
        print('entity_num:', entity_num) if debug else None

        # generate the mask for transformer
        mask = torch.arange(0, self.max_entities).float()
        mask = mask.repeat(batch_size, 1)
        mask = mask < entity_num.unsqueeze(dim=1)

        print('mask:', mask) if debug else None
        print('mask.shape:', mask.shape) if debug else None

        # mask: [batch_size, max_entities]
        device = next(self.parameters()).device
        mask = mask.to(device)

        # assert the input shape is : batch_seq_size x entities_size x embeding_size
        # note: because the feature size of entity is not equal to 256, so it can not fed into transformer directly.
        # thus, we add a embedding layer to transfer it to right size.
        # x is batch_entities_tensor (dim = 3). Shape: batch_seq_size x entities_size x embeding_size
        x = self.embedd(x)
        print('x.shape:', x.shape) if debug else None

        # mask for transformer need a special format
        mask_seq_len = mask.shape[-1]
        tran_mask = mask.unsqueeze(1)

        # tran_mask: [batch_seq_size x max_entities x max_entities]
        tran_mask = tran_mask.repeat(1, mask_seq_len, 1)

        # out: [batch_seq_size x entities_size x embeding_size]
        out = self.transformer(x, mask=tran_mask)
        print('out.shape:', out.shape) if debug else None

        # entity_embeddings: [batch_seq_size x entities_size x conv1_output_size]
        entity_embeddings = F.relu(self.conv1(F.relu(out).transpose(1, 2))).transpose(1, 2)
        print('entity_embeddings.shape:', entity_embeddings.shape) if debug else None

        # AlphaStar: The mean of the transformer output across across the units (masked by the missing entries) 
        # is fed through a linear layer of size 256 and a ReLU to yield `embedded_entity`
        masked_out = out * mask.unsqueeze(2)

        # sum over across the units
        # masked_out: [batch_seq_size x entities_size x embeding_size]
        # z: [batch_size, embeding_size]
        z = masked_out.sum(dim=1, keepdim=False)

        # here we should dived by the entity_num, not the cls.max_entities
        # z: [batch_size, embeding_size]
        z = z / entity_num

        # note, dim=1 means the mean is across all entities in one timestep
        # The mean of the transformer output across across the units  
        # is fed through a linear layer of size 256 and a ReLU to yield `embedded_entity`
        # embedded_entity: [batch_size, fc1_output_size]
        embedded_entity = F.relu(self.fc1(z))
        print('embedded_entity.shape:', embedded_entity.shape) if debug else None

        return entity_embeddings, embedded_entity


class Entity(object):

    def __init__(self, unit_type=1,
                 unit_attributes=[0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0], alliance=0,
                 health=10, shield=20, energy=50,
                 cargo_space_taken=0, cargo_space_max=0, build_progress=0,
                 current_health_ratio=0.4, current_shield_ratio=0.5, current_energy_ratio=0.7,
                 health_max=100, shield_max=50, energy_max=40,
                 display_type=1, x=123, y=218, is_cloaked=3, is_powered=True, is_hallucination=False, is_active=True,
                 is_on_screen=True, is_in_cargo=False, current_minerals=1000, current_vespene=1500, mined_minerals=500,
                 mined_vespene=300, assigned_harvesters=8, ideal_harvesters=14, weapon_cooldown=5.0, orders=[0, 1, 3, 0],
                 attack_upgrade_level=2, armor_upgrade_level=1, shield_upgrade_level=0, is_selected=True, is_targeted=False,
                 order_length=4, order_id_0=1, order_id_1=0, order_id_2=3, order_id_3=2, order_progress_0=50, 
                 order_progress_1=95, buff_id_0=12, buff_id_1=8, addon_unit_type=4, tag=0):
        super().__init__()
        self.unit_type = unit_type
        self.unit_attributes = unit_attributes
        self.alliance = alliance
        self.health = health
        self.shield = shield
        self.energy = energy
        self.cargo_space_taken = cargo_space_taken
        self.cargo_space_max = cargo_space_max
        self.build_progress = build_progress
        self.current_health_ratio = current_health_ratio
        self.current_shield_ratio = current_shield_ratio
        self.current_energy_ratio = current_energy_ratio
        self.health_max = health_max
        self.shield_max = shield_max
        self.energy_max = energy_max
        self.display_type = display_type
        self.x = x
        self.y = y 
        self.is_cloaked = is_cloaked
        self.is_powered = is_powered
        self.is_hallucination = is_hallucination
        self.is_active = is_active
        self.is_on_screen = is_on_screen
        self.is_in_cargo = is_in_cargo
        self.current_minerals = current_minerals
        self.current_vespene = current_vespene
        self.mined_minerals = mined_minerals
        self.mined_vespene = mined_vespene
        self.assigned_harvesters = assigned_harvesters
        self.ideal_harvesters = ideal_harvesters
        self.weapon_cooldown = weapon_cooldown
        self.attack_upgrade_level = attack_upgrade_level
        self.armor_upgrade_level = armor_upgrade_level
        self.shield_upgrade_level = shield_upgrade_level
        self.is_selected = is_selected
        self.is_targeted = is_targeted
        self.order_length = order_length
        self.order_id_1 = order_id_0
        self.order_id_2 = order_id_1
        self.order_id_3 = order_id_2
        self.order_id_4 = order_id_3
        self.order_progress_1 = order_progress_0
        self.order_progress_2 = order_progress_1
        self.buff_id_1 = buff_id_0
        self.buff_id_2 = buff_id_1
        self.addon_unit_type = addon_unit_type
        self.tag = tag

    def __str__(self):
        return 'unit_type: ' + str(self.unit_type) + ', alliance: ' + str(self.alliance) + ', health: ' + str(self.health)


def benchmark(e_list):
    # benchmark test
    benchmark_start = time.time()

    for i in range(1000):
        entities_tensor = torch.tensor(EntityEncoder.preprocess_numpy(e_list))

    elapse_time = time.time() - benchmark_start
    elapse_time = datetime.timedelta(seconds=elapse_time)
    print("Preprocess time {}".format(elapse_time))

    # benchmark test
    benchmark_start = time.time()

    # for i in range(1000):
    #     entities_tensor = EntityEncoder.preprocess_in_tensor(e_list)

    elapse_time = time.time() - benchmark_start
    elapse_time = datetime.timedelta(seconds=elapse_time)
    print("Preprocess time {}".format(elapse_time))

    print('entities_tensor:', entities_tensor) if debug else None
    print('entities_tensor.shape:', entities_tensor.shape) if debug else None
    # entities_tensor (dim = 2): entities_size x embeding_size


def test(debug=False):
    print(torch.tensor(np.unpackbits(np.array([25], np.uint8))))
    batch_size = 10

    e_list = []
    e1 = Entity(115, [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0], 0, 100, 60, 50, 4, 8, 95, 0.2, 0.0, 0.0, 140, 60, 100,
                1, 123, 218, 3, True, False, True, True, False, 0, 0, 0, 0, 0, 0, 3.0, [2, 3], 2, 1, 0, True, False)
    e2 = Entity(1908, [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0], 2, 1500, 0, 200, 0, 4, 15, 0.5, 0.8, 0.5, 1500, 0, 250,
                2, 69, 7, 3, True, False, False, True, False, 0, 0, 0, 0, 10, 16, 0.0, [1], 1, 1, 0, False, False)
    e_list.append(e1)
    e_list.append(e2)

    encoder = EntityEncoder()

    # test use preproces in numpy array
    entities_tensor = torch.tensor(EntityEncoder.preprocess_numpy(e_list))

    # test use preproces in torch tensor
    # entities_tensor = EntityEncoder.preprocess_in_tensor(e_list)

    entities_tensor = entities_tensor.unsqueeze(0)

    batch_entities_tensor = entities_tensor

    for i in range(batch_size):
        entities_tensor_copy = entities_tensor.detach().clone()
        batch_entities_tensor = torch.cat([batch_entities_tensor, entities_tensor_copy], dim=0)   

    print('batch_entities_tensor:', batch_entities_tensor) if 1 else None    
    print('batch_entities_tensor.shape:', batch_entities_tensor.shape) if 1 else None

    entity_embeddings, embedded_entity = encoder.forward(batch_entities_tensor)

    print('entity_embeddings.shape:', entity_embeddings.shape) if debug else None
    print('embedded_entity.shape:', embedded_entity.shape) if debug else None

    if debug:
        print("This is a test!")


if __name__ == '__main__':
    test()
