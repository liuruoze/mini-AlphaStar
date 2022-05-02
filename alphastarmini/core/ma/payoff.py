#!/usr/bin/env python
# -*- coding: utf-8 -*-

" The payoff matrix for all players (including agents) in the league, now it also contains all the players"

# modified from AlphaStar pseudo-code

import collections

import numpy as np

from alphastarmini.core.ma.player import Player

__author__ = "Ruo-Ze Liu"

debug = False


class Payoff:

    def __init__(self):
        self._players = []
        self._wins = collections.defaultdict(lambda: 0)
        self._draws = collections.defaultdict(lambda: 0)
        self._losses = collections.defaultdict(lambda: 0)
        self._games = collections.defaultdict(lambda: 0)
        self._decay = 0.99

    def _win_rate(self, _home, _away):
        if self._games[_home, _away] == 0:
            return 0.5

        return (self._wins[_home, _away]
                + 0.5 * self._draws[_home, _away]) / self._games[_home, _away]

    def __getitem__(self, match):
        home, away = match

        if isinstance(home, Player):
            home = [home]
        if isinstance(away, Player):
            away = [away]

        win_rates = np.array([[self._win_rate(h, a) for a in away] for h in home])
        if win_rates.shape[0] == 1 or win_rates.shape[1] == 1:
            win_rates = win_rates.reshape(-1)

        return win_rates

    def update(self, home, away, result):
        for stats in (self._games, self._wins, self._draws, self._losses):
            stats[home, away] *= self._decay
            stats[away, home] *= self._decay

        self._games[home, away] += 1
        self._games[away, home] += 1
        if result == "win":
            self._wins[home, away] += 1
            self._losses[away, home] += 1
        elif result == "draw":
            self._draws[home, away] += 1
            self._draws[away, home] += 1
        else:
            self._wins[away, home] += 1
            self._losses[home, away] += 1

    def add_player(self, player):
        self._players.append(player)

    def get_players_num():
        return len(self._players)

    @property
    def players(self):
        return self._players
