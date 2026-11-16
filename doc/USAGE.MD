# Usage

**English** | [中文](USAGE_CHS.MD)

## Can't find map

Supervised learning can fail when it opens a replay:

```
Failed to open the map archive: /home/StarCraftII/Battle.net\Cache/fd/88/...
```

The cause is that SC2 cannot find the map. Run these two commands in [this Colab](https://colab.research.google.com/drive/1TzO2Wi9KLjfBZeOqjGgjlxmwU5IX4wIV#scrollTo=XoZu1wZWLrfP):

```bash
!wget http://blzdistsc2-a.akamaihd.net/Linux/SC2.4.1.2.60604_2018_05_16.zip
!unzip -P iagreetotheeula SC2.4.1.2.60604_2018_05_16.zip -d ~ && rm -rf SC2.4.1.2.60604_2018_05_16.zip
```

The download fills the cache with maps; copy them to the path in the error message.

## Running reinforcement learning

The entry point is `rl_train_with_replay.test` in `run.py`. Three things matter.

**Self-play.** Two mAS agents play each other, one learning and one from a historical snapshot, so
training starts two SC2 processes. Playing SC2 against itself needs SC2 >= 4.0 and PySC2 >= 3.0;
the 4.1.2 build used above is enough, and PySC2 3.0 is already a dependency.

**The z reward.** mAS does not train on the game result alone: it also uses the z reward, computed
from build order and unit-count statistics, which is read from replays. That adds a third SC2
process running the replay, stepped in sync with the two game processes.

**Two versions side by side.** The replay may be older than the game (4.1.2 for the game, 3.16.1
for the replay), so several SC2 versions must run at once. SC2 supports this: each version sits
under `Versions` in the SC2 directory.

The Colab archive above contains both 3.16.1 (Base55958) and 4.1.2 (Base60321).
