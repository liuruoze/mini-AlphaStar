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
from alphastarmini.core.arch import baseline

from alphastarmini.core.sl import load_pickle

from alphastarmini.core.rl import action
from alphastarmini.core.rl import env_utils
from alphastarmini.core.rl import actor
from alphastarmini.core.rl import against_computer
from alphastarmini.core.rl import pseudo_reward

if __name__ == '__main__':
    # if we don't add this line, it may cause running time error while in Windows
    torch.multiprocessing.freeze_support()

    print("run init")

    #from alphastarmini.core.sl import transform_replay_data
    # transform_replay_data.test(on_server=False)

    #from alphastarmini.core.sl import sl_train_by_pickle
    # sl_train_by_pickle.test(on_server=False)

    # from alphastarmini.core.rl import rl_train_with_replay
    # rl_train_with_replay.test(on_server=False)

    # from alphastarmini.core.rl import rl_train_wo_replay
    # rl_train_wo_replay.test(on_server=False)

    # against_computer.test(on_server=False)

    #from alphastarmini.core.sl import analyze_replay_statistic
    # analyze_replay_statistic.test(on_server=False)

    print('run over')
