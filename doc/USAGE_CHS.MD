# 用法
> **Abstract (English).** Two answers: (1) "Can't find map" during supervised learning means the map
> cache is missing — the two commands below fetch it; (2) running reinforcement learning needs
> SC2 >= 4.0 and PySC2 >= 3.0, plus several SC2 versions side by side (game processes and a replay
> process). Full English text: [`USAGE.MD`](USAGE.MD).

这里给出 mini-AlphaStar（简称 mAS）的一些实用指南。

> 本文件是 [`USAGE.MD`](USAGE.MD) 的中文镜像，内容与英文版一一对应。

## 关于"找不到地图（Can't find map?）"

做监督学习时，打开回放可能遇到下面的报错：

"Failed to open the map archive: /home/StarCraftII/Battle.net\Cache/fd/88/..."

这是 SC2 找不到地图导致的。可以在下面这个地址里执行两条命令
"https://colab.research.google.com/drive/1TzO2Wi9KLjfBZeOqjGgjlxmwU5IX4wIV#scrollTo=XoZu1wZWLrfP"：

!wget http://blzdistsc2-a.akamaihd.net/Linux/SC2.4.1.2.60604_2018_05_16.zip
!unzip -P iagreetotheeula SC2.4.1.2.60604_2018_05_16.zip -d ~ && rm -rf SC2.4.1.2.60604_2018_05_16.zip

这一步会把大量地图同时下载到缓存里，之后只需要把它们复制到报"找不到地图"的那个位置即可。

## 关于"怎么跑强化学习（How to run reinforcement learning?）"

要跑强化学习（例如运行 `run.py` 里的 `rl_train_with_replay.test`），需要先了解几点。

**第一，mAS 的强化学习是自对弈训练的**，即两个 mAS 智能体互相对打（一个是正在学习的智能体，
另一个是历史快照里的智能体）。因此训练时 mAS 会启动**两个 SC2 进程**，每个进程对应一个智能体。
但 SC2 的 API 要支持互相对打，有如下要求：**SC2 版本 >= 4.0、PySC2 >= 3.0**。
所以要跑 mAS 的强化学习，请准备好对应版本的 SC2（可以直接用前面监督学习里提到的 SC2 4.1.2），
并下载 PySC2 3.0（它本来就是 mAS 的依赖之一）。

**第二，与其他强化学习不同，mAS 的训练需要一个特殊奖励——z 奖励**
（来自 SC2 里建造顺序与单位数量的统计）。这个 z 奖励来自回放。
因此每次训练时，mAS 还会启动一个**回放进程**（一个跑着看回放的 SC2 进程）。
回放进程与两个对局进程同步推进，学习过程从回放进程里收集 z 奖励。

然而回放的版本可能与我们跑的对局进程不同（例如对局跑 4.1.2，而回放是 3.16.1），
所以我们必须**同时运行多个不同版本的 SC2**。SC2 本身支持这个机制：
在 SC2 目录下有一个 `Versions` 目录，存放多个版本的 SC2。
只要 3.16.1（Base55958）与 4.1.2（Base60321）都在该目录里，就可以同时运行它们。

做法与前面一样：在
"https://colab.research.google.com/drive/1TzO2Wi9KLjfBZeOqjGgjlxmwU5IX4wIV#scrollTo=XoZu1wZWLrfP"
下载 SC2，这个版本已经同时包含 3.16.1 与 4.1.2。这样就可以直接开始强化学习了。
