# 用法

[English](USAGE.MD) | **中文**

## 找不到地图（Can't find map）

监督学习打开回放时可能报错：

```
Failed to open the map archive: /home/StarCraftII/Battle.net\Cache/fd/88/...
```

原因是 SC2 找不到地图。在[这个 Colab](https://colab.research.google.com/drive/1TzO2Wi9KLjfBZeOqjGgjlxmwU5IX4wIV#scrollTo=XoZu1wZWLrfP) 里执行两条命令：

```bash
!wget http://blzdistsc2-a.akamaihd.net/Linux/SC2.4.1.2.60604_2018_05_16.zip
!unzip -P iagreetotheeula SC2.4.1.2.60604_2018_05_16.zip -d ~ && rm -rf SC2.4.1.2.60604_2018_05_16.zip
```

这一步会把地图下载到缓存，再把缓存里的地图复制到报错信息给出的路径即可。

## 运行强化学习（How to run reinforcement learning）

入口是 `run.py` 里的 `rl_train_with_replay.test`。有三点需要先知道。

**自对弈。** 两个 mAS 智能体互相对打，一个是正在学习的，一个是历史快照，因此训练会启动两个
SC2 进程。SC2 之间对局要求 SC2 ≥ 4.0、PySC2 ≥ 3.0；上面用过的 4.1.2 即可，PySC2 3.0 本来
就是 mAS 的依赖。

**z 奖励。** mAS 不只用胜负做奖励，还用到 z 奖励（由建造顺序与单位数量统计得到），它从回放里
读。因此训练会再启动一个跑回放的 SC2 进程，与两个对局进程同步推进。

**多版本并存。** 回放版本可能与对局不同（例如对局 4.1.2、回放 3.16.1），需要同时运行多个版本。
SC2 支持这一点：各版本放在 SC2 目录的 `Versions` 下即可并存。

上面那个 Colab 包同时含 3.16.1（Base55958）与 4.1.2（Base60321），下载后可直接开始训练。
