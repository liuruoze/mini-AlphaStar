import os
import csv
import shutil
import numpy as np
import pandas as pd

from tqdm import tqdm

from absl import app
from absl import flags

from pysc2 import run_configs
from pysc2.lib import features
from pysc2.lib import point

from s2clientprotocol import sc2api_pb2 as sc_pb

FLAGS = flags.FLAGS
flags.DEFINE_bool("render", True, "Whether to render with pygame.")
flags.DEFINE_bool("realtime", False, "Whether to run in realtime mode.")
flags.DEFINE_bool("full_screen", False, "Whether to run full screen.")

flags.DEFINE_float("fps", 22.4, "Frames per second to run the game.")
flags.DEFINE_integer("step_mul", 5, "Game steps per observation.")
flags.DEFINE_bool("render_sync", False, "Turn on sync rendering.")
flags.DEFINE_integer("screen_resolution", 64,
                     "Resolution for screen feature layers.")
flags.DEFINE_integer("minimap_resolution", 64,
                     "Resolution for minimap feature layers.")

flags.DEFINE_integer("max_game_steps", 0, "Total game steps to run.")
flags.DEFINE_integer("max_episode_steps", 0, "Total game steps per episode.")

flags.DEFINE_bool("disable_fog", True, "Disable fog of war.")
flags.DEFINE_integer("observed_player", 1, "Which player to observe.")

flags.DEFINE_bool("save_data", False, "replays_save data or not")
flags.DEFINE_string("save_path", "./data/new_data/", "path to replays_save replay data")

ON_SERVER = False

if ON_SERVER:
    REPLAY_PATH = os.environ.get("SC2PATH") + "/Replays/filtered_replays_1/"
    COPY_PATH = None
    SAVE_PATH = "./result.csv"
else:
    REPLAY_PATH = os.environ.get("SC2PATH") + "/Replays/filtered_replays_1/"
    COPY_PATH = None
    SAVE_PATH = "./result.csv"

RACE = ['Terran', 'Zerg', 'Protoss', 'Random']
RESULT = ['Victory', 'Defeat', 'Tie']


def check_info(replay_info):
    map_name = replay_info.map_name
    player1_race = replay_info.player_info[0].player_info.race_actual
    player2_race = replay_info.player_info[1].player_info.race_actual

    print('map_name:', map_name)
    print('player1_race:', player1_race)
    print('player2_race:', player2_race)

    return True


def store_info(replay_info):
    map_name = replay_info.map_name
    player1_race = RACE[replay_info.player_info[0].player_info.race_requested - 1]
    player2_race = RACE[replay_info.player_info[1].player_info.race_requested - 1]
    game_duration_loops = replay_info.game_duration_loops
    game_duration_seconds = replay_info.game_duration_seconds
    game_version = replay_info.game_version
    game_result = RESULT[replay_info.player_info[0].player_result.result - 1]
    return [map_name,
            game_version,
            game_result,
            player1_race,
            player2_race,
            game_duration_loops,
            game_duration_seconds]


def main(argv):
    run_config = run_configs.get(version="3.16.1")
    print('REPLAY_PATH:', REPLAY_PATH)
    replay_files = os.listdir(REPLAY_PATH)
    print('length of replay_files:', len(replay_files))

    result = []
    map_set = set()

    screen_resolution = point.Point(32, 32)
    minimap_resolution = point.Point(32, 32)
    camera_width = 24
    random_seed = 42

    interface = sc_pb.InterfaceOptions(
        raw=True, score=True,
        feature_layer=sc_pb.SpatialCameraSetup(width=camera_width))
    screen_resolution.assign_to(interface.feature_layer.resolution)
    minimap_resolution.assign_to(interface.feature_layer.minimap_resolution)

    with run_config.start(full_screen=False) as controller:
        for replay_file in tqdm(replay_files):
            try:
                replay_path = REPLAY_PATH + replay_file
                print('replay_path:', replay_path)
                replay_data = run_config.replay_data(replay_path)
                replay_info = controller.replay_info(replay_data)

                start_replay = sc_pb.RequestStartReplay(
                    replay_data=replay_data,
                    options=interface,
                    disable_fog=FLAGS.disable_fog,
                    observed_player_id=FLAGS.observed_player)

                print(" Replay info ".center(60, "-"))
                print(replay_info)
                print("-" * 60)

                print("------start_replay")
                controller.start_replay(start_replay)
                print("------feature_layer")
                #feature_layer = features.Features(controller.game_info())
                #
                feature_layer = features.features_from_game_info(game_info=controller.game_info())

                print("------end feature_layer")
                frame_num = replay_info.game_duration_loops
                print("frame_num:", frame_num)
                step_num = frame_num // FLAGS.step_mul
                print("step_num:", step_num)
                path = FLAGS.save_path

                # init data
                player_data = np.zeros((step_num, 1 + 11))
                unit_data = np.zeros((step_num, 1 + 7))
                score_data = np.zeros((step_num, 1 + 13))

                frame_array = [(x + 1) * FLAGS.step_mul for x in range(step_num)]
                player_data[:, 0] = unit_data[:, 0] = score_data[:, 0] = frame_array

                order_data = np.array([])

                print("------controller.observe()")
                while True:
                    o = controller.observe()

                    try:
                        obs = feature_layer.transform_obs(o)

                        if o.player_result:  # end of game
                            print(o.player_result)
                            break

                        if o.actions:
                            # pass
                            func = feature_layer.reverse_action(o.actions[0])
                            print('func:', func)

                    except Exception as inst:
                        print(type(inst))
                        print(inst.args)
                        print(inst) 

                    controller.step()

                # We only test the first one replay            
            except Exception as inst:
                print(type(inst))
                print(inst.args)
                print(inst) 

                break

                continue

            break

            map_set.add(replay_info.map_name)

            if check_info(replay_info):
                print('check right!', replay_file)
                result.append([replay_file] + store_info(replay_info))
                #shutil.copy(replay_path, COPY_PATH)

    df = pd.DataFrame(result, columns=['Replay File', 'Map Name', 'Game Version', 'Game Result', 'Player1 Race', 'Player2 Race', 'Game Loops', 'Game Duration'])
    df.to_csv(path_or_buf=SAVE_PATH)

    print(map_set)


if __name__ == '__main__':
    app.run(main)
