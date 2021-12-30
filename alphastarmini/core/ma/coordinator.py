#!/usr/bin/env python
# -*- coding: utf-8 -*-

" The code for the coordinator in the league training"

# from AlphaStar pseudo-code

import numpy as np

__author__ = "Ruo-Ze Liu"

debug = False


class Coordinator:
    """Central worker that maintains payoff matrix and assigns new matches."""

    def __init__(self, league, output_file=None):
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
        self.output_file = output_file

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

        win_rate = self.results[2] / (1e-9 + total_episodes)

        statistic = 'Avg: [{}/{}] | Bot Difficulty: {} | win_rate: {:.1f} | food_used: {:.1f} | army_count: {:.1f} | std(army_count): {:.1f} | collected_points: {:.1f} | used_points: {:.1f} | killed_points: {:.1f} | steps: {:.3f} | Total time: {:.3f}s \n'.format(
            total_episodes, MAX_EPISODES, self.difficulty, win_rate, np.mean(self.food_used_list), np.mean(self.army_count_list), np.std(self.army_count_list), np.mean(self.collected_points_list),
            np.mean(self.used_points_list), np.mean(self.killed_points_list), np.mean(self.steps_list), np.mean(self.total_time_list))

        print("statistic: ", statistic)

        with open(self.output_file, 'a') as file:      
            file.write(statistic)
