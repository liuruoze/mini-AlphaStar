＃Usage

We give some useful guides for mini-AlphaStar (short as mAS) here.

## Question about "Can't find map?"

If you are doing supervised learning, when opening replay and encounter the following errors:

"Failed to open the map archive: /home/StarCraftII/Battle.net\Cache/fd/88/..."

This is caused by the SC2 cannot find the map, you can execute the following two commands in this url
"https://colab.research.google.com/drive/1TzO2Wi9KLjfBZeOqjGgjlxmwU5IX4wIV#scrollTo=XoZu1wZWLrfP":

!wget http://blzdistsc2-a.akamaihd.net/Linux/SC2.4.1.2.60604_2018_05_16.zip
!unzip -P iagreetotheeula SC2.4.1.2.60604_2018_05_16.zip -d ~ && rm -rf SC2.4.1.2.60604_2018_05_16.zip

This operation will download a lot of maps to the cache at the same time, and then you only need to copy it to the place where the map is not found.


## Question about "How to run reinforcement learning?"

To run reinforcement learning, e.g., run the "rl_train_with_replay.test" in "run.py", you need to know some basic notes. 

First, mAS's reinforcement learning is trained by self-play, i.e., two mAS agents will play against each other (one is the learning agent, the other is an agent from history snapshot).
So, when training, mAS will start two SC2 processes, each for one agent. However, the SC2 API for fighting against each other need following requirements: SC2 version >= 4.0 and PySC2 >= 3.0.
Hence, to run mAS's reinforcement learning, prepare the right versions of SC2 (you can directly use the SC2 4.1.2 mentioned before in supervised learning), and download PySC2 3.0 (already is a requirement for mAS).

Second, different from other reinforcement learning, mAS's training needs a special reward, which is called z reward (coming from build order and unit counts statistics in SC2). This z reward comes from replays.
So, in each training, mAS will also start a replay process (an SC2 process, which is running to see the replays). The replays process and the 2 game process steps together. And the learning process is collected z reward from the replay process.
However, the replay version may not be the same version as we running the game process (e.g., we run the game of version 4.1.2, and the replay is in 3.16.1), so we have to run multi different versions of SC2 at the same time.
This also can be done due to SC2 support this mechanism. In the dir of SC2, you can see a "Versions" dir, which stores multi versions of SC2. If the 3.16.1 (Base55958) and 4.1.2 (Base60321) are all in that dir, you can run them meanwhile.
How to do that, this is the same way as in the before process, download the SC2 version in "https://colab.research.google.com/drive/1TzO2Wi9KLjfBZeOqjGgjlxmwU5IX4wIV#scrollTo=XoZu1wZWLrfP". This version of SC2 already contains 3.16.1 and 4.1.2.
So we can do the reinforcement learning right now.

