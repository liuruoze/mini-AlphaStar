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
Batch_Scale = 16
Seq_Scale = 16
Select_Scale = 4

handle_cuda_error = False

skip_entity_list = False
skip_autoregressive_embedding = False
