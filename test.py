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
from alphastarmini.core.rl import pseudo_reward
from alphastarmini.core.rl import rl_algo
from alphastarmini.core.rl import rl_loss

if __name__ == '__main__':
    print("test init")

    entity_encoder.test()
    scalar_encoder.test()
    spatial_encoder.test()

    action_type_head.test()
    delay_head.test()
    queue_head.test()
    selected_units_head.test()
    target_unit_head.test()
    location_head.test()

    action.test()
    pseudo_reward.test()
    baseline.test()

    arch_model.test()
    agent.test()
    rl_algo.test()
    rl_loss.test()

    print('test over')
