#!/usr/bin/env python
# -*- coding: utf-8 -*-

" The code for the coordinator in the league training"

# from AlphaStar pseudo-code

import numpy as np

import torch

__author__ = "Ruo-Ze Liu"

debug = False


class Coordinator:
    """Central worker that maintains payoff matrix and assigns new matches."""

    def __init__(self, league, winrate_scale, output_file=None, results_lock=None, 
                 writer=None):
        self.league = league

        self.food_used_list = []
        self.army_count_list = []
        self.collected_points_list = []
        self.used_points_list = []
        self.killed_points_list = []
        self.steps_list = []

        self.total_time_list = []

        self.results = [0, 0, 0]
        self.difficulty = 1

        self.results_lock = results_lock
        self.output_file = output_file
        self.writer = writer

        self.winrate_scale = winrate_scale

    def set_uninitialed_results(self, actor_nums, episode_nums):
        self.actor_nums = actor_nums
        self.episode_nums = episode_nums
        self.episode_points = np.ones([actor_nums, episode_nums], dtype=np.float) * (-1e9)
        self.episode_outcome = np.ones([actor_nums, episode_nums], dtype=np.float) * (-1e9)

    def send_episode_points(self, agent_idx, episode_nums, results): 
        episode_id = episode_nums - 1
        self.episode_points[agent_idx - 1, episode_id] = results
        single_episode_points = self.episode_points[:, episode_id]
        if not (single_episode_points == (-1e9)).any():
            self.writer.add_scalar('coordinator/points', np.mean(single_episode_points), episode_id + 1)

    def send_episode_outcome(self, agent_idx, episode_nums, results): 
        episode_id = episode_nums - 1
        self.episode_outcome[agent_idx - 1, episode_id] = results
        single_episode_outcome = self.episode_outcome[:, episode_id]
        if not (single_episode_outcome == (-1e9)).any():
            self.writer.add_scalar('coordinator/outcome', np.mean(single_episode_outcome), episode_id + 1)
            update_id = int(episode_id / 2)
            self.update_winrate(update_id)

    def update_winrate(self, update_id):
        scale = self.winrate_scale
        if self.episode_nums >= 2:
            episode_winrate = np.transpose(self.episode_outcome).reshape([int(self.episode_nums / scale), int(self.actor_nums * scale)])
            single_episode_outcome = episode_winrate[update_id]
            if not (single_episode_outcome == (-1e9)).any():
                win_rate = (single_episode_outcome == 1).sum() / len(single_episode_outcome)
                self.writer.add_scalar('coordinator/winrate', win_rate, update_id + 1)
            del episode_winrate

    def send_outcome(self, home_player, away_player, outcome):
        self.league.update(home_player, away_player, outcome)

        if home_player.ready_to_checkpoint():
            # is the responsibility of the coordinator to add player in the league
            # actually it is added to to the payoff
            self.league.add_player(home_player.checkpoint())

    def only_send_outcome(self, home_player, outcome):
        self.results[outcome + 1] += 1

    def send_eval_results(self, player, difficulty, food_used_list, army_count_list, collected_points_list, used_points_list, killed_points_list, steps_list, total_time):
        self.food_used_list.extend(food_used_list)
        self.army_count_list.extend(army_count_list)
        self.collected_points_list.extend(collected_points_list)
        self.used_points_list.extend(used_points_list)
        self.killed_points_list.extend(killed_points_list)
        self.steps_list.extend(steps_list)

        self.total_time_list.append(total_time)
        self.difficulty = difficulty

    def write_eval_results(self):
        print("write_eval_results: ", self.results)

        total_episodes = sum(self.results)
        MAX_EPISODES = total_episodes

        loss_num = self.results[0]
        tie_num = self.results[1]
        win_num = self.results[2]

        win_rate = self.results[2] / (1e-9 + total_episodes)

        statistic = 'Avg: [{}/{}] | Bot Difficulty: {} | Lose: {} | Tie: {} | Win: {} | win_rate: {:.3f} | food_used: {:.1f} | army_count: {:.1f} | std(army_count): {:.1f} | collected_points: {:.1f} | used_points: {:.1f} | killed_points: {:.1f} | steps: {:.3f} | Total time: {:.3f}s \n'.format(
            total_episodes, MAX_EPISODES, self.difficulty, loss_num, tie_num, win_num, win_rate, np.mean(self.food_used_list), np.mean(self.army_count_list), np.std(self.army_count_list), np.mean(self.collected_points_list),
            np.mean(self.used_points_list), np.mean(self.killed_points_list), np.mean(self.steps_list), np.mean(self.total_time_list))

        print("statistic: ", statistic)

        with open(self.output_file, 'a') as file:      
            file.write(statistic)
