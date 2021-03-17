import alphastarmini
import torch

from alphastarmini.core.arch import entity_encoder
from alphastarmini.core.arch import scalar_encoder
from alphastarmini.core.arch import spatial_encoder
from alphastarmini.core.arch import arch_model
from alphastarmini.core.arch import action_type_head
from alphastarmini.core.arch import selected_units_head
from alphastarmini.core.arch import target_unit_head
from alphastarmini.core.arch import delay_head
from alphastarmini.core.arch import queue_head
from alphastarmini.core.arch import location_head
from alphastarmini.core.arch import agent

from alphastarmini.core.sl import transform_replay_data
from alphastarmini.core.sl import load_pickle
from alphastarmini.core.sl import sl_train_by_pickle

from alphastarmini.core.rl import action
from alphastarmini.core.rl import baseline
from alphastarmini.core.rl import env_utils
from alphastarmini.core.rl import actor
from alphastarmini.core.rl import rl_train

from alphastarmini.core.ma import vs_computer

from alphastarmini.lib import edit_distance

if __name__ == '__main__':
    torch.multiprocessing.freeze_support()
    print("run init")

    '''
    entity_encoder.test()
    scalar_encoder.test()
    spatial_encoder.test()

    action_type_head.test()
    delay_head.test()
    queue_head.test()
    selected_units_head.test()
    target_unit_head.test()
    location_head.test()
    '''

    # arch_model.test()
    # agent.test()
    # sc2_env.test(on_server=False)
    # actor.test(on_server=False)
    # baseline.test()

    # action.test()
    # transform_replay_data.test(on_server=False)
    # load_feature_label.test()
    # load_pickle.test()

    # sl_train_by_pickle.test(on_server=False)

    # rl_train.test(on_server=False)

    vs_computer.test(on_server=False)

    # edit_distance.test()
    # 
    # 

    print('run over')
