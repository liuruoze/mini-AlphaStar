"""Pseudocode for supervised training."""

import numpy as np
import tensorflow as tf

import human_data
from multiagent import Agent

BATCH_SIZE = 512
TRAJECTORY_LENGTH = 64
BO_PROBABILITY = 0.8
BU_PROBABILITY = 0.5
FINE_TUNING = False
RACES = ["Protoss", "Terran", "Zerg"]


def sample_random_batch(batch_size, agent_idx, mmr_cutoff):
  """Sample trajectories randomly from human replay source."""

  trajectories = []
  while len(trajectories) < batch_size:
    replay = get_random_trajectory(source=human_data,
                                   home_race=RACES[agent_idx],
                                   away_race=RACES,
                                   replay_filter=mmr_cutoff,
                                   filter_repeated_camera_moves=True)
    # Extract build order and build units vectors from replay
    bo = replay.get_BO(replay.home_player)
    bu = replay.get_BU(replay.home_player)
    # Sample Boolean variables bool_BO and bool_BU
    bool_bo = np.float(np.random.rand() < BO_PROBABILITY)
    bool_bu = np.float(np.random.rand() < BU_PROBABILITY)
    # Generate masked build order and build units
    masked_bo = bool_bo * bo
    masked_bu = bool_bu * bu
    z = [masked_bo, masked_bu]
    trajectory = (replay, z)
    trajectories.append(trajectory)
  return trajectories


def supervised_update(agent, optimizer, trajectories):
  """Update the agent parameters based on the losses."""

  parameters = agent.get_weights()
  # Compute the forward pass for the window
  policy_logits, _ = agent.unroll(trajectories)
  # Define MLE loss
  mle_loss = tf.nn.softmax_cross_entropy_with_logits(
      logits=policy_logits, labels=trajectories[0].target_policy)
  # Define L2 regularization loss
  l2_loss = (tf.reduce_sum([tf.nn.l2_loss(weight) for weight in parameters]))

  loss = mle_loss + 1e-5 * l2_loss
  agent.set_weights(optimizer.minimize(loss))


class Learner:
  """Learner worker that updates agent parameters based on trajectories.

  Uses a 128 core TPUv3 slice.
  """

  def __init__(self, index, race):
    self.agent_idx = index
    self.race = race
    self.mmr_cutoff = 3500 if not FINE_TUNING else 6200
    learning_rate = 1e-3 if not FINE_TUNING else 1e-5
    self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                            beta1=0.9,
                                            beta2=0.999,
                                            epsilon=1e-8)

  def update_parameters(self):
    trajectories = sample_random_batch(BATCH_SIZE,
                                       self.agent_idx,
                                       self.mmr_cutoff)
    # We remember the final LSTM state at the end of each trajectory and reuse
    # it when learning from the following trajectory.
    supervised_update(self.agent, self.optimizer, trajectories)

  @background
  def run(self):
    # Initialize the agent's weights.
    if not FINE_TUNING:
      self.agent = Agent(self.race, initialize_weights())
    else:
      self.agent = Agent(self.race, get_supervised_weights(self.race))
    while True:
      self.update_parameters()


def main():
  for agent_idx in range(3):
    learner = Learner(agent_idx, RACES[agent_idx])
    learner.run()


if __name__ == "__main__":
  main()
