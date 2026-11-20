# param for some configs, for ease use of changing different servers
# also ease of use for experiments

'''whether is running on server, on server meaning use GPU with larger memoary'''
on_server = False
#on_server = True

'''The replay path'''
replay_path = "data/Replays/filtered_replays_1/"
#replay_path = "/home/liuruoze/data4/mini-AlphaStar/data/filtered_replays_1/"
#replay_path = "/home/liuruoze/mini-AlphaStar/data/filtered_replays_1/"

'''The mini scale used in hyperparameter'''
import os as _os

# 2026-11-20：可用环境变量覆盖（默认仍为 16，行为完全不变），用于扫 batch / seq 规模。
# batch_size = 16*8/Batch_Scale，sequence_length = 16*8/Seq_Scale
Batch_Scale = int(_os.environ.get("MAS_BATCH_SCALE", "16"))
Seq_Scale = int(_os.environ.get("MAS_SEQ_SCALE", "16"))
Select_Scale = 4

handle_cuda_error = False

skip_entity_list = False
skip_autoregressive_embedding = False
