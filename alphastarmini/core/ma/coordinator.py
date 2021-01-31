#!/usr/bin/env python
# -*- coding: utf-8 -*-

" The code for the coordinator in the league training"

# from AlphaStar pseudo-code

__author__ = "Ruo-Ze Liu"

debug = False


class Coordinator:
    """Central worker that maintains payoff matrix and assigns new matches."""

    def __init__(self, league):
        self.league = league

    def send_outcome(self, home_player, away_player, outcome):
        self.league.update(home_player, away_player, outcome)

        if home_player.ready_to_checkpoint():
            # is the responsibility of the coordinator to add player in the league
            # actually it is added to to the payoff
            self.league.add_player(home_player.checkpoint())
